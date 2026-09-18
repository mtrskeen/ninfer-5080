# README snippet — what can be published

Short, defensible table for the fork README (RTX 5080 16 GB, this revision and
artifact, `--kv-dtype i4 --spec mtp --draft-tokens 3`, prefix reuse off, greedy
or the model's registered stochastic sampler as noted). Full methodology and
raw records: [BENCHMARKS.md](../../docs/BENCHMARKS.md).

## Recommended 16 GB profile

```text
--max-context 124928 --kv-capacity 124928 --kv-dtype i4
--spec mtp --draft-tokens 3 --prefill-chunk 256 --no-cuda-graph
```

`--prefill-chunk 256` replaces the previous `64` and is the single largest
improvement: prefill 594 → 1 500 tok/s at 32K and 433 → 1 017 tok/s at 123K
(2.35–2.53×), TTFT at 123K 284 → 121 s, for +34 MiB. Draft window 3 is optimal;
MTP2 is slower despite higher acceptance.

Chunk 256 is not bit-neutral (can change outputs at near-ties) and lowers the
absolute capacity boundary to 125,952; the safe ceiling stays 124,928.
`--lm-head-draft` does not fit at 124,928.

## Publishable numbers (MTP3, RTX 5080 16 GB)

| prompt tokens | prefill tok/s (chunk 256) | TTFT | decode tok/s | peak VRAM |
|---:|---:|---:|---:|---:|
| 32,714 | 1,500 | 22 s | 143 (NIAH short answer) | 15,858 MiB |
| 97,873 | 1,116 | 88 s | 128 (NIAH short answer) | 15,858 MiB |
| 122,861 | 1,017 | 121 s | 129 (NIAH short answer) | 15,858 MiB |

(8K at chunk 64 is 666 tok/s prefill, TTFT 11.6 s; chunk 256 was not separately
measured at 8K, so no 8K chunk-256 figure is claimed.)

Decode depends strongly on content (MTP3, stochastic): structured output
144 tok/s at 97% acceptance (3.90 tok/round); Python 128 tok/s at 82%; story
~71 tok/s at 31%; long reasoning 80–84 tok/s at 40–43%. MTP0 is a flat
~53 tok/s. A single decode number for this model is misleading; quote a range
and the workload.

## Safe ceilings

| mode | safe context | absolute test result |
|---|---:|---|
| text + MTP3 | **124,928** | 128,000 starts (427 MiB free); OOM between 128,000 and 129,024 |
| text MTP0 | **131,072** | 131,072 fits (~733 MiB free) |
| Vision + MTP3 | **32,768** | hard Vision merged-token envelope; startup fits to ~81,920 |

## Compatibility wording

- The same `sm_120a` build and artifact are `expected compatible, not
  hardware-tested` on RTX 5060 Ti 16 GB (`sm_120`, 16,384 MiB). Do not claim
  measured support. Decode speed will be lower due to lower memory bandwidth
  and fewer SMs; VRAM ceilings should transfer.

## Not to be published as facts

- Any RTX 5090 comparison; weights and configurations differ.
- Bit-exact agreement of greedy MTP3 output with a reference BF16 model (only
  cross-configuration near-tie divergence was characterized).
- Absolute long-run thermal stability.
- RULER scores (harness metadata override did not apply; use the direct NIAH
  table or rerun with NVIDIA RULER).
