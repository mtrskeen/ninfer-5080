
### mtp3

| ctx (prompt tokens) | samples | prefill tok/s | server TTFT (s) | decode tok/s | MTP accept | tokens/round | peak VRAM (MiB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7680 | 5 | 665.9 | 11.55 | 149.1 | 1.000 | 4.00 | 15824 |
| 32714 | 5 | 593.8 | 55.15 | 142.7 | 1.000 | 4.00 | 15824 |
| 64512 | 5 | 524.9 | 123.01 | 138.1 | 1.000 | 4.00 | 15824 |
| 97873 | 3 | 468.0 | 209.24 | 127.9 | 1.000 | 4.00 | 15824 |
| 122861 | 2 | 433.0 | 283.93 | 128.7 | 1.000 | 4.00 | 15824 |

### mtp0

| ctx (prompt tokens) | samples | prefill tok/s | server TTFT (s) | decode tok/s | MTP accept | tokens/round | peak VRAM (MiB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7680 | 5 | 665.9 | 11.55 | 53.4 | — | — | 15474 |
| 32714 | 5 | 596.6 | 54.89 | 51.1 | — | — | 15474 |
| 64512 | 5 | 527.1 | 122.51 | 49.2 | — | — | 15474 |
| 97873 | 3 | 469.6 | 208.58 | 45.2 | — | — | 15474 |
| 122861 | 2 | 434.5 | 282.94 | 45.7 | — | — | 15474 |

### mtp3-stochastic

| fixture | n | completion tok | prefill tok/s | decode tok/s | MTP accept | tokens/round | finish |
|---|---:|---:|---:|---:|---:|---:|---|
| decode_prose | 3 | 2023 | 461.0 | 84.4 | 0.424 | 2.27 | stop_token |
| long_decode_aime26_01 | 5 | 2031 | 577.2 | 112.2 | 0.678 | 3.03 | stop_token |
| long_decode_aime26_15 | 5 | 8192 | 599.9 | 80.3 | 0.396 | 2.19 | output_limit |
| long_decode_aime26_30 | 5 | 8192 | 607.2 | 84.2 | 0.432 | 2.29 | output_limit |
| scenario_code_cuda | 5 | 4096 | 525.3 | 116.5 | 0.719 | 3.16 | output_limit |
| scenario_code_python | 5 | 4096 | 554.3 | 127.7 | 0.821 | 3.46 | output_limit |
| scenario_code_typescript | 5 | 4096 | 552.7 | 118.8 | 0.740 | 3.22 | output_limit |
| scenario_story_en_mystery | 5 | 4096 | 590.1 | 82.7 | 0.414 | 2.24 | output_limit |
| scenario_story_zh_dialogue | 5 | 3796 | 561.4 | 71.1 | 0.309 | 1.93 | output_limit+stop_token |
| scenario_story_zh_scifi | 5 | 3112 | 575.7 | 71.2 | 0.309 | 1.93 | stop_token |
| scenario_structured_csv | 5 | 4096 | 534.8 | 136.7 | 0.902 | 3.71 | output_limit |
| scenario_structured_jsonl | 5 | 4096 | 585.2 | 143.9 | 0.968 | 3.90 | output_limit |
| scenario_structured_sql | 5 | 3780 | 486.3 | 134.3 | 0.880 | 3.64 | output_limit+stop_token |
| scenario_translation_en_zh | 5 | 898 | 627.4 | 116.3 | 0.714 | 3.14 | stop_token |
| scenario_translation_markdown | 5 | 825 | 617.6 | 110.5 | 0.660 | 2.98 | stop_token |
| scenario_translation_zh_en | 5 | 926 | 615.1 | 126.7 | 0.807 | 3.42 | stop_token |

### mtp3-greedy

| fixture | n | completion tok | prefill tok/s | decode tok/s | MTP accept | tokens/round | finish |
|---|---:|---:|---:|---:|---:|---:|---|
| long_decode_aime26_01 | 1 | 3275 | 574.9 | 118.4 | 0.735 | 3.21 | stop_token |
| long_decode_aime26_15 | 1 | 8192 | 601.5 | 91.2 | 0.494 | 2.48 | output_limit |
| long_decode_aime26_30 | 1 | 8192 | 604.3 | 97.5 | 0.551 | 2.65 | output_limit |
| scenario_code_cuda | 1 | 4096 | 523.5 | 122.3 | 0.767 | 3.30 | output_limit |
| scenario_code_python | 1 | 4096 | 549.5 | 132.8 | 0.866 | 3.60 | output_limit |
| scenario_code_typescript | 1 | 4096 | 549.7 | 129.1 | 0.833 | 3.50 | output_limit |
| scenario_story_en_mystery | 1 | 4096 | 588.7 | 101.0 | 0.578 | 2.74 | output_limit |
| scenario_story_zh_dialogue | 1 | 3685 | 554.5 | 81.1 | 0.399 | 2.20 | stop_token |
| scenario_story_zh_scifi | 1 | 3142 | 569.3 | 79.6 | 0.384 | 2.15 | stop_token |
| scenario_structured_csv | 1 | 4096 | 538.9 | 138.2 | 0.915 | 3.74 | output_limit |
| scenario_structured_jsonl | 1 | 4096 | 584.8 | 146.6 | 0.991 | 3.97 | output_limit |
| scenario_structured_sql | 1 | 4096 | 486.3 | 141.9 | 0.947 | 3.84 | output_limit |
| scenario_translation_en_zh | 1 | 868 | 630.7 | 124.3 | 0.782 | 3.35 | stop_token |
| scenario_translation_markdown | 1 | 852 | 617.7 | 112.8 | 0.683 | 3.05 | stop_token |
| scenario_translation_zh_en | 1 | 1009 | 612.3 | 132.2 | 0.855 | 3.57 | stop_token |

### mtp0-greedy

| fixture | n | completion tok | prefill tok/s | decode tok/s | MTP accept | tokens/round | finish |
|---|---:|---:|---:|---:|---:|---:|---|
| long_decode_aime26_01 | 1 | 1679 | 585.3 | 53.7 | — | — | stop_token |
| long_decode_aime26_15 | 1 | 8192 | 607.7 | 53.2 | — | — | output_limit |
| long_decode_aime26_30 | 1 | 8192 | 612.6 | 53.2 | — | — | output_limit |
| scenario_code_cuda | 1 | 4096 | 532.5 | 53.5 | — | — | output_limit |
| scenario_code_python | 1 | 4096 | 564.1 | 53.5 | — | — | output_limit |
| scenario_code_typescript | 1 | 4096 | 559.6 | 53.5 | — | — | output_limit |
| scenario_story_en_mystery | 1 | 3834 | 598.4 | 53.5 | — | — | stop_token |
| scenario_story_zh_dialogue | 1 | 3395 | 571.9 | 53.5 | — | — | stop_token |
| scenario_story_zh_scifi | 1 | 3153 | 587.0 | 53.6 | — | — | stop_token |
| scenario_structured_csv | 1 | 4096 | 548.6 | 53.5 | — | — | output_limit |
| scenario_structured_jsonl | 1 | 4096 | 594.8 | 53.5 | — | — | output_limit |
| scenario_structured_sql | 1 | 4096 | 496.7 | 53.5 | — | — | output_limit |
| scenario_translation_en_zh | 1 | 817 | 629.7 | 53.7 | — | — | stop_token |
| scenario_translation_markdown | 1 | 747 | 622.6 | 53.7 | — | — | stop_token |
| scenario_translation_zh_en | 1 | 1507 | 619.5 | 53.7 | — | — | stop_token |

### parameter sweep

| block | spec | draft | kv | chunk | graph | lm-head | prefill tok/s | decode tok/s | MTP accept | tokens/round | peak VRAM |
|---|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|
| chunk-128 | mtp | 3 | i4 | 128 | False | None | 1162.1 | 142.7 | 1.000 | 4.00 | 15834 |
| chunk-256 | mtp | 3 | i4 | 256 | False | None | 1499.9 | 142.6 | 1.000 | 4.00 | 15858 |
| chunk-64 | mtp | 3 | i4 | 64 | False | None | 593.8 | 142.6 | 1.000 | 4.00 | 15824 |
| spec-mtp0 | none | 0 | i4 | 64 | False | None | 471.8 | 54.0 | — | — | 15474 |
| spec-mtp1 | mtp | 1 | i4 | 64 | False | None | 464.2 | 78.5 | 0.768 | 1.77 | 15820 |
| spec-mtp2 | mtp | 2 | i4 | 64 | False | None | 463.9 | 89.3 | 0.613 | 2.23 | 15822 |
| spec-mtp3 | mtp | 3 | i4 | 64 | False | None | 461.9 | 96.8 | 0.528 | 2.58 | 15824 |
| spec-mtp4 | mtp | 4 | i4 | 64 | False | None | 459.0 | 68.0 | 0.389 | 2.55 | 15826 |
| spec-mtp5 | mtp | 5 | i4 | 64 | False | None | 454.5 | 67.2 | 0.341 | 2.70 | 15828 |

### MTP window sweep

| block | fixture | n | completion tok | decode tok/s | MTP accept | tokens/round | peak VRAM |
|---|---|---:|---:|---:|---:|---:|---:|
| mtp0-long | decode_prose | 1 | 1024 | 53.6 | — | — | 13988 |
| mtp0-long | gen_long | 3 | 2048 | 53.7 | — | — | 13988 |
| mtp1-long | decode_prose | 1 | 1024 | 74.3 | 0.684 | 1.68 | 14242 |
| mtp1-long | gen_long | 3 | 2048 | 74.4 | 0.685 | 1.68 | 14242 |
| mtp2-long | decode_prose | 1 | 1024 | 83.9 | 0.554 | 2.11 | 14244 |
| mtp2-long | gen_long | 3 | 2048 | 82.3 | 0.532 | 2.06 | 14244 |
| mtp3-long | decode_prose | 1 | 1024 | 84.1 | 0.423 | 2.27 | 14246 |
| mtp3-long | gen_long | 3 | 2048 | 86.2 | 0.442 | 2.32 | 14246 |
| mtp4-long | decode_prose | 1 | 1024 | 64.2 | 0.358 | 2.43 | 14248 |
| mtp4-long | gen_long | 3 | 2048 | 60.8 | 0.325 | 2.30 | 14248 |
| mtp5-long | decode_prose | 1 | 1024 | 60.9 | 0.293 | 2.46 | 14250 |
| mtp5-long | gen_long | 3 | 2048 | 61.2 | 0.295 | 2.47 | 14250 |

### capacity boundary

| block | max_context | kv_capacity | started | peak VRAM | request tokens |
|---|---:|---:|---|---:|---:|
| mtp0-cap-124928 | 124928 | 124928 | yes | 15472 | 30 |
| mtp0-cap-126976 | 126976 | 126976 | yes | 15504 | 30 |
| mtp0-cap-128000 | 128000 | 128000 | yes | — | 30 |
| mtp0-cap-129024 | 129024 | 129024 | yes | 15538 | 30 |
| mtp0-cap-130048 | 130048 | 130048 | yes | 15554 | 30 |
| mtp0-cap-131072 | 131072 | 131072 | yes | 15570 | 30 |
| mtp3-cap-125952 | 125952 | 125952 | yes | 15842 | 30 |
| mtp3-cap-126976 | 126976 | 126976 | yes | — | 30 |
| mtp3-cap-128000 | 128000 | 128000 | yes | 15876 | 30 |

### stress VRAM (per block, request order)

- `nearfull-cycle`: n=4 VRAM peaks [15824, 15824, 15824, 15824]
- `reuse-sequential`: n=40 VRAM peaks [None, 14246, None, 14246, None, 14246, None, None, 14246, None, 14246, None, 14246, None, 14246, None, 14246, None, 14246, None, 14246, None, 14246, None, None, 14246, None, 14246, None, 14246, None, 14246, None, 14246, None, 14246, None, 14246, None, 14246]
