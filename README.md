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

## RTX 5060 Ti 16 GB and other 16 GB `sm_120` cards

The fork is validated on RTX 5080 16 GB. A third-party run (2026-09-20) on a
GeForce RTX 5060 Ti 16 GB with **this fork and the same public artifact** confirms
the card is compatible:

- stable text context **98,304 (96K)** with one worker — 124,928 OOMs there
  because the desktop/display takes ~440 MiB;
- MTP3 decode ~**32 tok/s** on Russian prose (content-dependent — code/JSON is
  much faster), prefill ~**620 tok/s** at a 20K prompt;
- native vision in a **separate 32,768-token** server.

Full report (Russian, as received, with our verification status):
[validation/third-party/5060ti-20260920.md](validation/third-party/5060ti-20260920.md).

Comparison across 16 GB setups. These are **different runtimes, harnesses and
content mixes**, shown for orientation only:

| Setup | GPU | Runtime / artifact | Text context | MTP3 decode | Vision | Source |
|---|---|---|---:|---:|---|---|
| This fork | RTX 5080 16 GB | `sm_120a` min-Q4 + MTP-Q4/Vision-Q4 | **124,928** | 71–144 tok/s | 32,768 | local validation |
| This fork (third-party) | RTX 5060 Ti 16 GB | same fork + artifact | 98,304 | ~32 tok/s prose | 32,768 (separate) | report, 2026-09-20 |
| [freedomNTD/ninfer-5060ti](https://github.com/freedomNTD/ninfer-5060ti) | RTX 5060 Ti 16 GB | aljazceru fork + aaaljaz min-Q4 (no MTP-Q4, no vision) | ~49,152 | 48.8 tok/s, short answers | — | their README |
| [ruwwww/ninfer-5060ti](https://github.com/ruwwww/ninfer-5060ti) (`rtx-5060ti`) | RTX 5060 Ti 16 GB | upstream-style; 27B numbers are RTX 5090-only | 262,144 (9B) | 119–132 tok/s (9B) | 9B only | their README |
| aljaz A5000 fork | RTX A5000 16 GB | sm_86 min-Q4 | 122,880 | 22.4 tok/s | — | local validation |

Against the closest sibling — `freedomNTD/ninfer-5060ti`, the same 27B min-Q4 on
a 5060 Ti — this fork plus its artifact reaches roughly **twice the context**
(96K vs ~49K) and adds native vision and an MTP-Q4 head. Its 48.8 tok/s is a
short-answer/structured figure (high MTP acceptance) and is not comparable to
the prose figure above, so no speed ranking is drawn from it. The
`ruwwww/ninfer-5060ti` branch publishes RTX 5060 Ti numbers only for its 9B
artifacts, so a 27B-on-5060-Ti comparison is not possible there.

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

- Validated on RTX 5080 16 GB. An RTX 5060 Ti 16 GB is confirmed compatible by a
  third-party run (see above); other `sm_120` 16 GB cards are **expected
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
