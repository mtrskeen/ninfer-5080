# NInfer — RTX 5080 16 GB fork

**Qwen3.8-27B at 124,928 tokens of context on a single consumer 16 GB Blackwell card.**

This fork extends [aljazceru/ninfer](https://github.com/aljazceru/ninfer), which
extends [Neroued/ninfer](https://github.com/Neroued/ninfer). It adds a Blackwell
(`sm_120a`) port of the groupwise-int execution paths, a 16 GB Qwen3.8-27B
artifact (min-Q4 text + **MTP-Q4** + **Vision-Q4**), a fast prefill profile, and
a measured, reproducible validation campaign.

- Artifact: [`mtrskeen/qwen3.8-27b-ninfer-minq4-mtpq4-visionq4-5080`](https://huggingface.co/mtrskeen/qwen3.8-27b-ninfer-minq4-mtpq4-visionq4-5080) (15.5 GB, NInfer-only)
- Detailed results: [docs/BENCHMARKS.md](docs/BENCHMARKS.md)
- Build & quantize recipe with rationale: [docs/BUILDING.md](docs/BUILDING.md)
- Raw measurements: [validation/5080-20260918](validation/5080-20260918)
- Русская версия: [README.ru.md](README.ru.md)

## What it does

- Single-request text serving up to 124,928 tokens with MTP3 speculative decoding.
- Vision (image/video) in a separate `--vision` mode capped at 32,768 tokens.
- 2.35–2.53× faster prefill than the previous default by using
  `--prefill-chunk 256`.
- Runs the whole model plus a 124,928-token INT4 KV pool in 15.9 GB of VRAM.

## Recommended launch (RTX 5080 16 GB)

```bash
ninfer-serve qwen3_8_27b_minq4_mtpq4_visionq4.ninfer \
  --model-id qwen3.8-27b-16gb \
  --host 0.0.0.0 --port 18100 \
  --max-context 124928 --kv-capacity 124928 \
  --kv-dtype i4 --spec mtp --draft-tokens 3 \
  --prefill-chunk 256 --no-cuda-graph
```

Vision mode (separate server): add `--vision` and set
`--max-context 32768 --kv-capacity 32768`.

## Results at a glance

Values are single-request, one persistent server, prefix reuse disabled. They
describe **different runtimes and GPUs** and are shown for orientation only —
see [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for methodology and raw data.

| Variant | Hardware | Runtime | Context | long-gen decode | Notes |
|---|---|---|---:|---:|---|
| Baseline llama.cpp | RTX 5080 16 GB (same host) | llama.cpp b10853, UD-IQ3_XXS + q8_0 KV | 131,072 | **33.9 tok/s** @128K, 58.7 @8K | no MTP, no native vision |
| aljaz A5000 fork | RTX A5000 Laptop 16 GB | NInfer sm_86, min-Q4 | 122,880 | **22.4 tok/s** (MTP3), 12.6 plain @131K | first 16 GB port |
| **this fork** | **RTX 5080 16 GB** | **NInfer sm_120a, min-Q4 + MTP-Q4, `i4` KV, chunk 256** | **124,928** | **71–144 tok/s** (MTP3), 53 MTP0 | native vision, chunk 256 prefill |

Same-hardware comparison (RTX 5080, llama.cpp vs this fork, long generation):

| Context | llama.cpp decode | this fork decode (MTP3) | this fork prefill |
|---:|---:|---:|---:|
| ~32K | ~51.5 tok/s | 117–133 tok/s (code/translation), 71–86 (prose) | 1,500 tok/s |
| ~96K | ~40 tok/s | ~80–112 tok/s | 1,116 tok/s |
| ~123K | 33.9 tok/s | ~86 tok/s (prose), up to 129 (short answers) | 1,017 tok/s |

Decode on this fork is strongly content-dependent (MTP acceptance 31–99 %).
Structured output reaches ~144 tok/s; prose/story is CPU-of-the-model-limited at
~71–86 tok/s. `gsm8k_cot` (8-shot, lm-evaluation-harness, 50 samples): **0.96**; RULER NIAH
retrieval at 8K/32K/64K: 3/3 each.

## Install

Build the runtime image (CUDA 13.1, `sm_120a`):

```bash
docker build -t ninfer-aljaz:build -f Dockerfile.aljaz .
docker build -t ninfer-mtpq4:build -f Dockerfile.mtpq4 .
```

Then follow [docs/BUILDING.md](docs/BUILDING.md) to convert
`Qwen/Qwen3.8-27B` into the min-Q4/MTP-Q4/Vision-Q4 artifact, or download the
prebuilt artifact from Hugging Face.

## Documentation

| Document | EN | RU |
|---|---|---|
| Overview | [README.md](README.md) | [README.ru.md](README.ru.md) |
| Full test results, methodology, raw data | [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | [docs/BENCHMARKS.ru.md](docs/BENCHMARKS.ru.md) |
| Build & quantize recipe, design rationale | [docs/BUILDING.md](docs/BUILDING.md) | [docs/BUILDING.ru.md](docs/BUILDING.ru.md) |

## Status and limitations

- Validated on RTX 5080 16 GB. Other `sm_120` 16 GB cards are **expected
  compatible, not hardware-tested**.
- Text + MTP3 safe ceiling is 124,928 tokens (OOM between 128,000 and 129,024);
  MTP0 reaches 131,072. With `--prefill-chunk 256` the ceiling is 124,928
  (absolute 125,952).
- Vision is limited to 32,768 merged tokens by the runtime and cannot be
  combined with the 124,928-token text profile.
- Greedy MTP3 output is bit-identical to MTP0 for short answers and can diverge
  on long generations at near-tie tokens; both remain coherent (see findings).
- `--prefill-chunk 256` is a performance profile: it can change exact token
  choices versus chunk 64 at near-ties (decode speed is unchanged).
- `--lm-head-draft` does not fit at the 124,928 ceiling and needs a smaller
  context.

## License and attribution

- Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), Apache-2.0.
- Runtime: [Neroued/ninfer](https://github.com/Neroued/ninfer).
- 16 GB min-Q4 conversion and sm_86 port: [aljazceru/ninfer](https://github.com/aljazceru/ninfer).
- Blackwell MTP-Q4/Vision-Q4, 16 GB profile and validation: this fork.

Apache-2.0. Upstream NInfer's own license and documentation remain in place.
