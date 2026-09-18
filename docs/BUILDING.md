# Building and quantizing the 16 GB Qwen3.8-27B artifact

This is the detailed recipe that turns `Qwen/Qwen3.8-27B` into the NInfer
artifact used by this fork, plus the reasoning behind each decision. The
prebuilt artifact is published at
[`mtrskeen/qwen3.8-27b-ninfer-minq4-mtpq4-visionq4-5080`](https://huggingface.co/mtrskeen/qwen3.8-27b-ninfer-minq4-mtpq4-visionq4-5080)
(`qwen3_8_27b_minq4_mtpq4_visionq4.ninfer`, 15,514,935,296 bytes, sha256
`c7eb6fdde74bfd70e168a9ccce113b08fb34ad7ce3349ea989590c4830591fb8`).

## 0. Constraints that drive the design

A 27B model has to fit, together with a usable KV cache and speculative and
vision heads, into **16,303 MiB** of GPU memory:

| Component | Budget |
|---|---|
| Text weights | as small as quality allows |
| MTP draft head | must fit alongside text |
| Vision tower | must fit, but only when `--vision` is used |
| KV pool | as large as possible: context is the product |

The fork therefore reduces every large tensor to Q4 (`Q4G64_F16S`) and makes
Vision allocations lazy, keeping the KV pool as the main consumer of the
remaining VRAM.

## 1. Host requirements

- Linux x86_64, NVIDIA RTX 50-series (`sm_120`), driver >= 610.57
- Docker + NVIDIA Container Toolkit
- ~60 GB free disk (bf16 source ~56 GB + base + final artifact)
- Python 3.10+ with `torch`, `safetensors`, `numpy`, `transformers` for conversion

## 2. Build the runtime image

From the repository root (CUDA 13.1, `sm_120a`):

```bash
docker build -t ninfer-aljaz:build -f Dockerfile.aljaz .
docker build -t ninfer-mtpq4:build -f Dockerfile.mtpq4 .
```

`Dockerfile.aljaz` uses `nvidia/cuda:13.1.2-devel-ubuntu24.04` and configures
CMake with `-DCMAKE_CUDA_ARCHITECTURES=120a`. `Dockerfile.mtpq4` rebuilds
`ninfer` and `ninfer-serve` from the current source on top of it. The campaign
image was `ninfer-mtpq4:build`
`sha256:723834522026fb327a54b04aed7b6c9a6421b19e4596677e85efba7a9981445a`.

## 3. Download the source checkpoint

```bash
huggingface-cli download Qwen/Qwen3.8-27B --local-dir Qwen3.8-27B
```

The converter verifies the official SHA-256 of the six frontend resources
(`tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`,
`generation_config.json`, `preprocessor_config.json`,
`video_preprocessor_config.json`).

## 4. Convert to the min-Q4 base artifact

The fork's `tools/convert/qwen3_8_27b/inventory.py` overrides the registered
layout with the minimal all-Q4 layout:

- MLP, GDN/attention input and output projections -> Q4;
- fused Q4/Q4 input kernels for GDN and attention;
- Q4 `linear_add` for residuals;
- Q4 output head;
- token embedding stays Q6.

```bash
python3 -m tools.convert.qwen3_8_27b.convert \
  --model Qwen3.8-27B \
  --out out/qwen3_8_27b_minq4.ninfer \
  --device cuda
```

Output: `out/qwen3_8_27b_minq4.ninfer` (~15.1 GB). MTP and Vision are still W8/Q5
at this point.

**Why this layout.** The quality gate is a simulated-PPL evaluator
(`tools/eval/sim_ppl.py`) that applies the fork's exact groupwise quantization
math and measures LM loss on Wikitext-2 (50x512) against the IQ3_XXS anchor
(PPL 6.2569). The all-Q4 layout passed at **+0.04 PPL**; a more aggressive
Q3-MLP variant failed at +0.79 and was rejected. Q6 is kept for the embedding
because there is no Q4 gather route and the embedding is quality-sensitive.

## 5. Requantize MTP and Vision to Q4 (256-byte alignment)

Each tensor must be 256-byte aligned and the payload 4096-byte aligned;
otherwise the runtime aborts with `tensor is not 256-byte aligned`.

```bash
# 5a. MTP matrices -> Q4G64_F16S
python3 tools/artifact/repack_mtp_q4.py \
  --src out/qwen3_8_27b_minq4.ninfer \
  --dst out/qwen3_8_27b_minq4_mtpq4.ninfer

# 5b. Vision tower matrices -> Q4G64_F16S
python3 tools/artifact/repack_vision_q4.py \
  --src out/qwen3_8_27b_minq4_mtpq4.ninfer \
  --dst out/qwen3_8_27b_minq4_mtpq4_visionq4.ninfer
```

**Why repack MTP/Vision.** MTP and Vision are stored W8/Q5 in the registered
profile. Quantizing them to Q4 frees several hundred MiB that go straight into
the KV pool, which is what buys the 124,928-token context. The repackers only
touch those tensors; the text body is carried over byte-for-byte.

**Why lazy Vision.** With eager allocation the Vision workspace consumes VRAM
even in text-only runs, lowering the text context ceiling. The fork allocates
and frees it on demand (`patch_vision_lazy.py` behavior), so text runs keep the
full pool and Vision pays only when used.

## 6. Verify

```bash
python3 tools/artifact/verify_artifacts.py out/qwen3_8_27b_minq4_mtpq4_visionq4.ninfer
bash tools/artifact/run_load_verify.sh out/qwen3_8_27b_minq4_mtpq4_visionq4.ninfer
```

The published artifact was additionally validated end-to-end: it loads in
~12 s, reports `target=qwen3_8_27b`, `weights_id=groupwise-int`, 1,115 tensors
and 7 resources, and passed the full serving campaign in
[`../validation/5080-20260918`](../validation/5080-20260918).

## 7. Run

```bash
docker run --rm --gpus all --network host \
  -v "$PWD/models:/ninfer-models:ro" \
  ninfer-mtpq4:build \
  ninfer-serve /ninfer-models/qwen3_8_27b_minq4_mtpq4_visionq4.ninfer \
  --model-id qwen3.8-27b-16gb \
  --host 0.0.0.0 --port 18100 \
  --max-context 124928 --kv-capacity 124928 \
  --kv-dtype i4 --spec mtp --draft-tokens 3 \
  --prefill-chunk 256 --no-cuda-graph
```

## 8. Why the serving defaults

Every default below was measured on the RTX 5080 16 GB; see
[BENCHMARKS.md](BENCHMARKS.md).

| Setting | Choice | Reason |
|---|---|---|
| `--prefill-chunk` | **256** (was 64) | 2.35–2.53× prefill, +34 MiB; 512 OOMs |
| `--spec mtp --draft-tokens` | **3** | throughput optimum; MTP2 higher acceptance but slower; >5 unsupported |
| `--kv-dtype` | **i4** (int4-group128) | fastest and highest acceptance; `int8`/`bf16` OOM at 64K |
| `--max-context`/`--kv-capacity` | **124928** | safe ceiling with margin; MTP3 OOM between 128,000 and 129,024 |
| `--no-cuda-graph` | on | CUDA Graphs gave no measurable benefit |
| `--no-prefix-reuse` | benchmarks only | every prompt is prefilled in full for honest numbers |
| `--vision` | separate mode | needs a smaller pool; runtime caps vision at 32,768 tokens |

## 9. Provenance

The published artifact was built from fork revision
`024b3ea4b91b67fdd75d8ca947e2a58a4258237b` plus the working-tree patch
(`validation/5080-20260918/fork-working-tree.patch`, diff sha256
`d7259b5ac59b8a35fef685c549ec268944e62472bb0f63dd8c6f46434e3d1a13`) on
image `ninfer-mtpq4:build` `sha256:723834522026fb327a54b04aed7b6c9a6421b19e4596677e85efba7a9981445a`.

Apache-2.0. Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
Runtime: [Neroued/ninfer](https://github.com/Neroued/ninfer). 16 GB min-Q4 and
sm_86 port: [aljazceru/ninfer](https://github.com/aljazceru/ninfer).
