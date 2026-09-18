# NInfer fork validation on RTX 5080 16 GB — Qwen3.8-27B min-Q4 + MTP-Q4 + Vision-Q4

This document records a from-scratch re-validation of the fork's 16 GB
Qwen3.8-27B artifact on a single GeForce RTX 5080 16 GB, carried out to decide
what can be published. It is deliberately not a copy of the old numbers:
padding, prefix reuse, sampling and warm-up are described, raw records are
committed under `validation/5080-20260918/`, and every table is
derived from those records.

Nothing here was cherry-picked; the baseline was re-measured and, where the
baseline was not optimal, the better configuration is reported.

## 1. Provenance

| Item | Value |
|---|---|
| Fork revision | `024b3ea4b91b67fdd75d8ca947e2a58a4258237b` (`feature/mtp-q4`) |
| Working tree | dirty; Blackwell MTP-Q4 / Vision-Q4 patches applied (16 files, +144/−49) |
| Working-tree `git diff` sha256 | `d7259b5ac59b8a35fef685c549ec268944e62472bb0f63dd8c6f46434e3d1a13` |
| Diff file | `validation/5080-20260918/fork-working-tree.patch` |
| Runtime image | `ninfer-mtpq4:build` = `sha256:723834522026fb327a54b04aed7b6c9a6421b19e4596677e85efba7a9981445a` |
| Artifact | `qwen3_8_27b_minq4_mtpq4_visionq4.ninfer`, 15,514,935,296 bytes |
| Artifact sha256 | `c7eb6fdde74bfd70e168a9ccce113b08fb34ad7ce3349ea989590c4830591fb8` |
| Build recipe | [docs/BUILDING.md](BUILDING.md) |

The working tree is not committed at the time of writing; the exact source is
reconstructible from the revision plus the diff above.

## 2. Hardware and software

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5080, 16,303 MiB, compute capability 12.0 |
| Driver | 610.57.04 |
| CUDA | 13.1.115 (compile and runtime, container `nvidia/cuda:13.1.2-devel-ubuntu24.04`) |
| Kernel | `6.12.107+deb13-amd64` |
| Docker | 29.8.0 (NVIDIA Container Toolkit, `--gpus all`) |
| Host | `validation host`, 24 vCPU, 62 GiB RAM |

## 3. Methodology

Baseline configuration under test:

```text
--max-context 124928 --kv-capacity 124928 --kv-dtype i4
--spec mtp --draft-tokens 3 --prefill-chunk 64 --no-cuda-graph
--max-concurrency 1 --default-max-tokens 4096 --no-prefix-reuse
```

Metrics are computed from the server's own `--request-log-jsonl` records and
match upstream `tools/bench/run_serve_corpus.py`:

```text
prefill_tok_s        = prompt_tokens / prefill_seconds
server_ttft_ms       = 1000 * (prepare_seconds + vision_seconds + prefill_seconds)
decode_tok_s         = (completion_tokens - 1) / decode_seconds
spec_acceptance      = accepted_tokens / drafted_tokens
spec_tokens_per_round= 1 + accepted_tokens / speculative_rounds
```

- Requests are serial (`stream=false`) to one persistent server per block.
- Each configuration block starts a fresh server; a text warm-up runs before
  measurement so cold start is separated from steady state (model load
  ~11.8–12.5 s, recorded per block in `server_start_*.json`).
- Prefix reuse is disabled (`--no-prefix-reuse`) so every prompt is prefilled
  in full.
- `peak VRAM` is sampled with `nvidia-smi` at 0.25 s during the request; it is
  the allocator-visible peak for the whole block (the KV pool is reserved at
  startup, so long prompts do not raise the peak).
- Greedy blocks pass `--greedy` (exact argmax). Stochastic blocks use either
  the model's registered sampler defaults (temperature 0.7, top-p 0.8, top-k 20,
  presence 1.5 — the "workloads" campaign) or the upstream published profile
  (temperature 0.6, top-p 0.95, top-k 20, presence 1.0 — the "mtpk" campaign).
  The difference is called out where it matters.
- Fixtures are the repository's `examples/cli/messages/*` set plus NIAH
  fixtures generated from the committed `long_niah_8k` body at 32K/96K/123K
  (the needle is inserted at the midpoint). Exact prompt token counts are read
  from the server log.

Raw data: `validation/5080-20260918/<campaign>/records.jsonl`
(one JSON object per measured request, including full phase timings, speculative
counters, VRAM peak and the server's exact configuration), plus `summary.json`,
`summary.csv` and per-request response bodies where they matter.

## 4. Context-length profile (greedy, NIAH, 128-token answers)

### MTP3 (`--spec mtp --draft-tokens 3`)

| prompt tokens | n | prefill tok/s | server TTFT (s) | decode tok/s | peak VRAM (MiB) |
|---:|---:|---:|---:|---:|---:|
| 7,680 | 5 | 665.9 | 11.55 | 149.1 | 15,824 |
| 32,714 | 5 | 593.8 | 55.15 | 142.7 | 15,824 |
| 64,512 | 5 | 524.9 | 123.01 | 138.1 | 15,824 |
| 97,873 | 3 | 468.0 | 209.24 | 127.9 | 15,824 |
| 122,861 | 2 | 433.0 | 283.93 | 128.7 | 15,824 |

### MTP0 (no speculation)

| prompt tokens | n | prefill tok/s | server TTFT (s) | decode tok/s | peak VRAM (MiB) |
|---:|---:|---:|---:|---:|---:|
| 7,680 | 5 | 665.9 | 11.55 | 53.4 | 15,474 |
| 32,714 | 5 | 596.6 | 54.89 | 51.1 | 15,474 |
| 64,512 | 5 | 527.1 | 122.51 | 49.2 | 15,474 |
| 97,873 | 3 | 469.6 | 208.58 | 45.2 | 15,474 |
| 122,861 | 2 | 434.5 | 282.94 | 45.7 | 15,474 |

Notes:

- Prefill is indistinguishable between MTP0 and MTP3; MTP costs nothing before
  the first decode step.
- Prefill throughput falls 666 → 433 tok/s between 8K and 123K prompts. The
  often-quoted ~620 tok/s only holds for short prompts.
- MTP3 decode at these lengths is 128–149 tok/s because the NIAH answer is only
  17 tokens and acceptance is 100%; this is **not** a representative decode
  figure. See section 5 for real generation.
- The peak is dominated by the resident KV pool (`kv-capacity`), not by prompt
  length. MTP3 costs ~350 MiB more than MTP0 (draft head + graph allowance) and
  therefore caps `kv-capacity` lower.

## 5. Content workloads (MTP3, stochastic, 5 seeds each where seeded)

Acceptance and decode speed are strongly content-dependent. The upstream
scenario corpus with the model's registered sampler defaults:

| fixture | n | completion tok | prefill tok/s | decode tok/s | MTP accept | tokens/round |
|---|---:|---:|---:|---:|---:|---:|
| scenario_structured_jsonl | 5 | 4,096 | 585.2 | 143.9 | 96.8% | 3.90 |
| scenario_structured_csv | 5 | 4,096 | 534.8 | 136.7 | 90.2% | 3.71 |
| scenario_structured_sql | 5 | 3,780 | 486.3 | 134.3 | 88.0% | 3.64 |
| scenario_code_python | 5 | 4,096 | 554.3 | 127.7 | 82.1% | 3.46 |
| scenario_code_typescript | 5 | 4,096 | 552.7 | 118.8 | 74.0% | 3.22 |
| scenario_translation_zh_en | 5 | 926 | 615.1 | 126.7 | 80.7% | 3.42 |
| scenario_translation_en_zh | 5 | 898 | 627.4 | 116.3 | 71.4% | 3.14 |
| scenario_code_cuda | 5 | 4,096 | 525.3 | 116.5 | 71.9% | 3.16 |
| scenario_translation_markdown | 5 | 825 | 617.6 | 110.5 | 66.0% | 2.98 |
| long_decode_aime26_01 | 5 | 2,031 | 577.2 | 112.2 | 67.8% | 3.03 |
| long_decode_aime26_30 | 5 | 8,192 | 607.2 | 84.2 | 43.2% | 2.29 |
| decode_prose (4K budget) | 3 | 2,023 | 461.0 | 84.4 | 42.4% | 2.27 |
| long_decode_aime26_15 | 5 | 8,192 | 599.9 | 80.3 | 39.6% | 2.19 |
| scenario_story_en_mystery | 5 | 4,096 | 590.1 | 82.7 | 41.4% | 2.24 |
| scenario_story_zh_scifi | 5 | 3,112 | 575.7 | 71.2 | 30.9% | 1.93 |
| scenario_story_zh_dialogue | 5 | 3,796 | 561.4 | 71.1 | 30.9% | 1.93 |

Single greedy run per fixture (for comparison, same fixtures):

| fixture | MTP3 decode / accept / tok/round | MTP0 decode |
|---|---:|---:|
| scenario_structured_jsonl | 146.6 / 99.1% / 3.97 | 53.5 |
| scenario_structured_csv | 138.2 / 91.5% / 3.74 | 53.5 |
| scenario_structured_sql | 141.9 / 94.7% / 3.84 | 53.5 |
| scenario_code_python | 132.8 / 86.6% / 3.60 | 53.5 |
| scenario_code_typescript | 129.1 / 83.3% / 3.50 | 53.5 |
| scenario_code_cuda | 122.3 / 76.7% / 3.30 | 53.5 |
| long_decode_aime26_01 | 118.4 / 73.5% / 3.21 | 53.7 |
| scenario_translation_zh_en | 132.2 / 85.5% / 3.57 | 53.7 |
| scenario_translation_markdown | 112.8 / 68.3% / 3.05 | 53.7 |
| scenario_story_zh_dialogue | 81.1 / 39.9% / 2.20 | 53.5 |
| long_decode_aime26_15 | 91.2 / 49.4% / 2.48 | 53.2 |
| long_decode_aime26_30 | 97.5 / 55.1% / 2.65 | 53.2 |

The previously quoted 106–120 tok/s is reproduced only for
structured/code/translation-style work; prose and long reasoning fall to
71–84 tok/s at ~30–44% acceptance. MTP0 is a flat ~53.5 tok/s regardless of
content, so MTP3 is a 1.3–2.7× win depending on workload.

## 6. Speculative exactness

### 6.1 Short deterministic answers

For all 16 greedy MTP0/MTP3 pairs in the NIAH context sweep (8K/32K/64K/96K,
same prompt and seed) the response `content_sha256` is identical.

### 6.2 Long generations

For long greedy generations the two configurations do **not** produce
byte-identical output: 12/12 non-empty scenario outputs differ, usually within
the first few hundred characters (e.g. a JSONL tag `slow` vs `fast`, different
plausible numbers in CSV, different wording in a story). Divergence occurs at
near-tie argmax positions and then cascades. Both branches remain coherent:

- structured JSONL: 61/62 lines valid JSON in both configs (last line truncated
  by the output budget), 62 lines each;
- CSV: 134 rows each, correct column count;
- Python: all complete fenced blocks parse with `ast.parse` in both.

### 6.3 Determinism

Repeating the same greedy request twice in the same configuration reproduces
the identical `sha256` for both MTP3 and MTP0 (2×2 fixtures). The divergence is
therefore a cross-configuration numerical effect of different batch geometry
(speculative verify vs plain decode), not run-to-run nondeterminism. Tokens
emitted through verification are still the target argmax for the geometry that
produced them; this is a near-tie sensitivity, not observed corruption or
repetition. It means "lossless speculative decoding" is only bit-exact for
outputs where no near-tie flip occurs.

## 7. Parameter sweep

### 7.1 MTP draft window (greedy `decode_prose`, 1,024 tokens)

| draft k | decode tok/s | MTP accept | tokens/round |
|---:|---:|---:|---:|
| 0 (no spec) | 54.0 | — | — |
| 1 | 78.5 | 76.8% | 1.77 |
| 2 | 89.3 | 61.3% | 2.23 |
| **3 (baseline)** | **96.8** | 52.8% | 2.58 |
| 4 | 68.0 | 38.9% | 2.55 |
| 5 | 67.2 | 34.1% | 2.70 |

### 7.2 MTP draft window (stochastic, upstream profile, `gen_long` 2,048 tokens, 3 seeds)

| draft k | decode tok/s | MTP accept | tokens/round |
|---:|---:|---:|---:|
| 0 | 53.7 | — | — |
| 1 | 74.4 | 68.5% | 1.68 |
| 2 | 82.3 | 53.2% | 2.06 |
| **3 (baseline)** | **86.2** | 44.2% | 2.32 |
| 4 | 60.8 | 32.5% | 2.30 |
| 5 | 61.2 | 29.5% | 2.47 |

Draft window 3 is the optimum on both workloads. MTP2 raises acceptance
(61%/53% vs 53%/44%) but loses more to per-round cost than it gains in
tokens/round, so it is **slower** than MTP3. `--draft-tokens` above 5 is
rejected by the server (`--spec mtp requires --draft-tokens in [1,5]`).

### 7.3 Prefill chunk

| chunk | 32K prefill | vs 64 | 96K prefill | 123K prefill | TTFT 32K | TTFT 96K | TTFT 123K | decode | peak VRAM |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | rejected: `must be a positive multiple of 64` | — | — | — | — | — | — | — | — |
| 64 (baseline) | 593.8 | 1.00× | 467.7 | 432.8 | 55.2 s | 209.4 s | 284.1 s | 142.6 | 15,824 |
| 128 | 1,162.1 | 1.96× | — | — | 28.2 s | — | — | 142.7 | 15,834 |
| **256** | **1,499.9** | **2.53×** | **1,115.8** | **1,016.6** | **21.8 s** | **87.9 s** | **121.1 s** | 142.6 | 15,858 |
| 512 | OOM at startup | — | — | — | — | — | — | — | — |
| 1024 | OOM at startup | — | — | — | — | — | — | — | — |

Prefill speedup spans 1.96× (64→128) and 2.35–2.53× (64→256) across prompt
lengths. Decode and MTP acceptance are unchanged; only TTFT and workspace move.

`--prefill-chunk 64` is a severe bottleneck: raising it to 256 roughly
**2.5×** prefill throughput and TTFT for +34 MiB of workspace. Longer-context
confirmation is in `validation/5080-20260918/sweep2/`.

Chunk size is **not numerically transparent**. The same greedy prompt at chunk
64 and chunk 256 produced different (but coherent) outputs in all three tested
fixtures (structured JSONL, story, prose): prefill kernel routing shifts
near-tie tokens and the difference cascades. Decode is identical. Treat chunk
256 as a performance profile, not a bit-identical reconfiguration.

### 7.4 KV dtype, CUDA Graph, lm-head draft

Greedy `decode_prose` (1,024 tokens), one seed:

| configuration | decode tok/s | MTP accept | tokens/round | peak VRAM | notes |
|---|---:|---:|---:|---:|---|
| **`i4` (baseline)** | **97.0** | 52.8% | 2.58 | 14,246 | best speed and acceptance |
| `i4-g64` | 92.1 | 48.6% | 2.45 | 14,264 | slightly worse than group-128 |
| `int8` | 90.6 | 47.1% | 2.41 | 14,808 | fits at 32K only |
| `bf16` | 93.2 | 49.7% | 2.49 | 14,230 | fits at 8K only; still worse than `i4` |
| `i4` + CUDA Graph | 96.7 | 52.8% | 2.58 | 14,246 | no gain over eager |
| `i4` + `--lm-head-draft` | 101.3 | 53.0% | 2.59 | 14,586 | +4.4% for +340 MiB |
| `i4`, no `--lm-head-draft` | 97.0 | 52.8% | 2.58 | 14,246 | control |

At `--kv-capacity 65536` both `bf16` and `int8` KV OOM at startup, so on 16 GB
the only usable KV types at 64K are `i4`/`i4-g64`. `i4` is also the fastest and
highest-acceptance option, opposite to the earlier fork note that recommended
`int8` for MTP. CUDA Graphs give no measurable benefit with this artifact, so
`--no-cuda-graph` is justified. `--lm-head-draft` is a small, real win but its
extra 340 MiB must be checked against the chosen context ceiling.

At the 124,928 ceiling `--lm-head-draft` does **not** fit: the engine requests
2,455,868,160 B of runtime capacity but only 2,133,294,080 B are available, so
it OOMs at startup. It is usable only at a reduced context.

## 8. Concurrency

Upstream's decode-saturation method (C workers released together, 1,024-token
generations of `decode_prose`, one persistent server):

| C | per-request context | kv pool | makespan (s) | aggregate decode tok/s | MTP accept | peak VRAM (MiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16,384 | 16,384 | 20.2 | 94.1 | 51.9% | 13,966 |
| 2 | 16,384 | 32,768 | 30.3 | 125.3 | 51.8% | 14,404 |
| 2 | 32,768 | 65,536 | 30.3 | 125.3 | 51.8% | 14,964 |
| 4 | 16,384 | 65,536 | 50.8 | 150.0 | 50.9% | 15,282 |
| 4 | 24,576 | 98,304 | 50.8 | 150.0 | 50.9% | 15,844 |

Four concurrent 32K sessions (131,072 tokens of shared KV) do **not** fit on
16 GB: the MTP3 pool OOMs above 128K (section 10). Practical concurrent shapes
on this card are up to C=4 at ≤24K per request.

## 9. Stress and regression

- **Server reuse without restart**: 40 sequential short requests, flat peak
  VRAM 14,246 MiB, no drift; server stays healthy.
- **Near-full context reuse**: in one MTP3 server, 122,861 → 80 → 97,873 →
  122,861 prompt cycle; prefill 432.6/467.7/432.8 tok/s, decode 128.7/111.3/
  127.9/128.7 tok/s, peak VRAM constant 15,824 MiB, all requests finish
  correctly. No VRAM growth across the cycle.
- **Vision→text→Vision** in one server: image, image, multi-image, video,
  mixed image+video and mixed multi-turn requests all succeed; see section 11.
- **Invalid/boundary inputs** (server with `--vision`, ctx 65,536): empty
  messages, missing/bad role, unknown model, zero/negative `max_tokens`,
  unknown content type, CLI-style `image` part, malformed JSON and an
  over-context prompt all return HTTP 400/404 with a specific code; `/health`
  stays 200 after every case. `max_tokens` far above the context is clamped and
  served (200). Details in `.../invalid/results.json`.
- **Long continuous generation**: 8,192-token MTP3 generations (reasoning and
  `gen_long`) complete without repetition collapse; prose long-generation is
  bounded by acceptance (section 5).
- One real limitation found: `--vision` at `--max-context 124928` OOMs at
  startup (vision fixed allocations + full KV pool); vision needs a smaller
  pool.

## 10. Context ceiling and OOM boundary

Startup capacity sweep (server must reserve the KV pool before serving):

| profile | largest capacity started | peak VRAM | first failure |
|---|---:|---:|---|
| text + MTP3 | 128,000 | 15,876 MiB (~427 MiB free) | 129,024: runtime reservation needs 2,493,505,280 B, only 2,490,334,208 B available |
| text + MTP3, chunk 256 | 125,952 | 15,876 MiB | 126,976: OOM at startup |
| text MTP0 | 131,072 | 15,570 MiB (~733 MiB free) | not reached in tested range |
| Vision + MTP3 | 81,920 (startup); effective context capped at 32,768 by the Vision merged-token envelope | — | 98,304 OOM at startup |

Interpretation:

- **Safe published maximum for text+MTP3: 124,928** (baseline; the next tested
  step 125,952 still starts, and 128,000 starts with 427 MiB free, but with no
  practical headroom for allocator thrash). Absolute OOM boundary is between
  128,000 and 129,024.
- **Text without speculation: 131,072 fits** (~733 MiB free) and is the
  practical ceiling; MTP0's smaller footprint buys exactly the classic 128K.
- **Vision+MTP3: use 32,768.** The Vision runtime has a hard 32,768
  merged-token envelope, so larger `--max-context` does not extend usable
  vision context; the server does accept and correctly serve images at 65,536,
  and startup fits up to ~81,920.

## 11. Vision

`--vision`, MTP3, 32,768 context, greedy, thinking disabled. Requests use
OpenAI `image_url` data URIs; the fork's `examples/cli` files use a CLI-only
`{type: image}` shape and must be converted for HTTP.

| task | prompt tokens | vision s | decode tok/s | result | graded |
|---|---:|---:|---:|---|---|
| image_chart | 428 | 0.02 | 123.4 | `NIFER VISION 731；3；左侧` | correct |
| image_natural | 417 | 0.02 | 110.3 | mailbox 24, sun on the right | correct |
| multi_image_compare | 531 | 0.03 | 114.0 | left 2; right 3; new star | correct |
| video_temporal | 1,191 | 0.05 | 60.6 | `红色圆形；3；是；9` | coherent |
| mixed_image_video | 1,575 | 0.07 | 94.6 | `NIFER-9` | coherent |
| mixed_multiturn | 1,608 | 0.07 | 151.4 | `24-9` | coherent |
| image_chart @65,536 | 428 | 0.02 | 123.4 | same correct answer | correct |

Vision preprocessing (`vision_seconds`) is 20–70 ms once the image is a data
URI; decode throughput stays in the normal MTP3 range. All three graded image
tasks (chart reading, scene attribute retrieval, two-image comparison) are
answered correctly, including Cyrillic/Chinese instruction following.

## 12. Standard external evaluation

`lm-evaluation-harness 0.4.13` against the same server (`--greedy
--no-thinking`, ctx 124,928, chunk 256), tokenizer `Qwen/Qwen3.8-27B`,
`openai-chat-completions` backend with `--apply_chat_template`:

| task | samples | metric | value |
|---|---:|---|---:|
| `gsm8k_cot` (8-shot) | 50 | exact_match (flexible/strict) | **0.96 ± 0.028** |

RULER NIAH (retrieval) through the same harness, 3 samples per length, greedy,
one length per run (`--metadata max_seq_lengths=[L]`; this harness takes
`key=value`, not a JSON string):

| length | niah_single_1 | niah_multikey_1 |
|---:|---:|---:|
| 8,192 | 1.00 | 1.00 |
| 32,768 | 1.00 | 1.00 |
| 65,536 | 1.00 | 1.00 |

Retrieval is perfect at every tested length. The direct NIAH sweep in section 4
extends this to 96K/123K prompts (also correct).

Logs: `validation/5080-20260918/raw/lmeval/` and
`.../raw/logs/`.

## 13. Portability audit — RTX 5060 Ti 16 GB (not hardware-tested)

The 5060 Ti is `sm_120` with 16,384 MiB, so the same `sm_120a` build is
expected to run; this has **not** been executed on real hardware. Code audit:

- Device gate is capability-based: `src/targets/qwen3_6/impl/runtime/layouts_impl.h:599`
  rejects only `sm() < 80`. There is no GPU-name whitelist.
- NVFP4/FP8 profiles are stubbed with loud runtime errors on non-`sm_89`/`sm_120a`
  builds (`src/ops/quant_disabled_stubs.cpp`,
  `src/ops/linear/nvfp4/nvfp4_w4a4_tma_stub.cpp`). The min-Q4 artifact used here
  is groupwise-int (q4/w8/bf16) and is unaffected.
- PDL is gated at runtime on compute capability major ≥ 9
  (`src/core/pdl.cuh`), correct on `sm_120`.
- Tuning constants `kRtx5090SmCount = 170` exist in
  `sparse_moe_prefill_kernels.cu` and `gated_delta_net/chunked/output.cu`. They
  only size grid/persistent-CTA heuristics and use grid-stride loops, so they
  affect performance, not correctness, on a 36–38 SM GB206 part. The 27B model
  is dense, so the sparse-MoE path is not exercised.
- No artificial 5080/5090-only check was found; unlike upstream, this fork does
  not reject non-`sm_120a` architectures at configure time
  (`CMakeLists.txt:9`).
- Performance expectations must not be transferred: 5060 Ti has ~448 GB/s vs
  ~960 GB/s memory bandwidth and fewer SMs, so decode will be slower at the
  same 16 GB footprint. VRAM/W capacity should transfer because the pool is
  sized from `totalGlobalMem` and the device has the same 16,384 MiB.

Permitted README wording: `expected compatible, not hardware-tested`.

## 14. Bugs, anomalies and new observations

1. **`--prefill-chunk 64` cripples prefill** (2.5× loss vs chunk 256) for +34 MiB.
2. **MTP draft window is capped at 5**, not 15 (`--spec mtp requires
   --draft-tokens in [1,5]`); chunk must be a multiple of 64.
3. **MTP3 vs MTP0 greedy output is not bit-identical on long generations**
   (near-tie flips), though short answers match and each configuration is
   internally deterministic.
4. **MTP acceptance is content-dependent**: 31–44% for prose/story vs 90–99%
   for structured output. A single headline acceptance number is misleading.
5. **`kv-capacity` sets the VRAM peak**, not prompt length; MTP3 costs ~350 MiB
   more than MTP0, which is why MTP0 reaches 128K and MTP3 stops at ~128K.
6. **`--vision` forces a much smaller KV pool** and is bounded by the 32,768
   merged-token Vision envelope; `--vision` at 124,928 OOMs at startup.
7. **HTTP vision uses `image_url`, not the CLI `{type: image}` shape**; the
   CLI fixtures must be converted for the API. (The earlier "modality not
   supported" result was a request-format error in the harness, not the fork.)
8. **`i4` KV is documented as `int4-group128`** by the server; only `i4`/`i4-g64`
   fit at 64K with MTP3, `int8`/`bf16` KV OOM at that capacity.
9. Upstream's `groupwise-int` profile is explicitly outside its published
   benchmark campaign; these results partly fill that gap for the 16 GB fork.
10. **`--prefill-chunk` is not bit-neutral**: chunk 64 vs 256 changes outputs at
    near-ties (3/3 greedy fixtures differed; all coherent).
11. **`--lm-head-draft` does not fit at 124,928** (OOM at startup); it needs a
    reduced context.
12. **`--prefill-chunk 256` lowers the absolute capacity boundary** to 125,952
    (from 128,000 at chunk 64) while keeping the safe ceiling 124,928.

## 15. Claims: proven vs not proven

Proven on this hardware (RTX 5080 16 GB, this revision and artifact):

- Prefill/decode/TTFT/acceptance/VRAM tables in sections 4–8, with raw records.
- `--prefill-chunk 256` is a ~2.5× prefill improvement that still fits.
- Draft window 3 is the throughput optimum; MTP2 is slower despite higher
  acceptance.
- Greedy short-answer speculative output is bit-identical; long greedy output
  can diverge at near-ties while remaining coherent.
- Safe text+MTP3 ceiling 124,928; text-MTP0 131,072; Vision effective 32,768.
- Vision works and is correct on chart/scene/comparison tasks.
- Error handling for malformed and over-context requests.
- External quality: gsm8k_cot 0.96 and RULER NIAH 3/3 at 8K/32K/64K (3 samples
  each).

Not proven:

- RTX 5060 Ti compatibility or performance (none tested).
- Absolute long-run thermal stability beyond the runs recorded here.
- Bit-exact token agreement with a reference BF16 target model (no logits
  comparison was performed).
- Anything about RTX 5090 or upstream profiles; configurations and weights
  differ and no cross-GPU claim is made.
