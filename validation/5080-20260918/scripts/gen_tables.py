#!/usr/bin/env python3
"""Generate markdown tables from validation campaign records."""
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

OUT = Path("/out")


def load(campaign):
    recs = []
    for line in (OUT / campaign / "records.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if "metrics" in r:
                recs.append(r)
    return recs


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def f(v, n=1):
    return "—" if v is None else f"{v:.{n}f}"


def ctx_table(recs, block):
    groups = defaultdict(list)
    order = ["long_niah_8k", "niah_32768", "long_niah_64k", "niah_98304", "niah_123392"]
    for r in recs:
        if r["block"] == block:
            groups[r["fixture"]].append(r)
    print(f"\n### {block}\n")
    print("| ctx (prompt tokens) | samples | prefill tok/s | server TTFT (s) | decode tok/s | MTP accept | tokens/round | peak VRAM (MiB) |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for fixture in order:
        rs = groups.get(fixture)
        if not rs:
            continue
        pt = rs[0]["metrics"]["prompt_tokens"]
        print(f"| {pt} | {len(rs)} | {f(mean([r['metrics']['prefill_tok_s'] for r in rs]))} | "
              f"{f(mean([r['metrics']['server_ttft_ms'] for r in rs]) / 1000, 2)} | "
              f"{f(mean([r['metrics']['decode_tok_s'] for r in rs]))} | "
              f"{f(mean([r['metrics']['spec_acceptance'] for r in rs]), 3)} | "
              f"{f(mean([r['metrics']['spec_tokens_per_round'] for r in rs]), 2)} | "
              f"{f(mean([r['vram_peak_mib'] for r in rs]), 0)} |")


def workload_table(recs):
    groups = defaultdict(list)
    for r in recs:
        groups[(r["block"], r["fixture"])].append(r)
    for block in ("mtp3-stochastic", "mtp3-greedy", "mtp0-greedy"):
        if not any(k[0] == block for k in groups):
            continue
        print(f"\n### {block}\n")
        print("| fixture | n | completion tok | prefill tok/s | decode tok/s | MTP accept | tokens/round | finish |")
        print("|---|---:|---:|---:|---:|---:|---:|---|")
        for k in sorted(groups, key=lambda x: (x[0], x[1])):
            if k[0] != block:
                continue
            rs = groups[k]
            m = [r["metrics"] for r in rs]
            print(f"| {k[1]} | {len(rs)} | {f(mean([x['completion_tokens'] for x in m]), 0)} | "
                  f"{f(mean([x['prefill_tok_s'] for x in m]))} | {f(mean([x['decode_tok_s'] for x in m]))} | "
                  f"{f(mean([x['spec_acceptance'] for x in m]), 3)} | "
                  f"{f(mean([x['spec_tokens_per_round'] for x in m]), 2)} | {'+'.join(sorted({str(x['finish_reason']) for x in m}))} |")


def sweep_table(recs):
    groups = defaultdict(list)
    for r in recs:
        groups[r["block"]].append(r)
    print("\n### parameter sweep\n")
    print("| block | spec | draft | kv | chunk | graph | lm-head | prefill tok/s | decode tok/s | MTP accept | tokens/round | peak VRAM |")
    print("|---|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for block in sorted(groups):
        rs = groups[block]
        cfg = rs[0]["config"]
        m = [r["metrics"] for r in rs]
        print(f"| {block} | {cfg.get('spec') or 'none'} | {cfg.get('draft_tokens',0)} | {cfg.get('kv_dtype')} | "
              f"{cfg.get('prefill_chunk')} | {cfg.get('cuda_graph')} | {cfg.get('lm_head_draft')} | "
              f"{f(mean([x['prefill_tok_s'] for x in m]))} | {f(mean([x['decode_tok_s'] for x in m]))} | "
              f"{f(mean([x['spec_acceptance'] for x in m]), 3)} | {f(mean([x['spec_tokens_per_round'] for x in m]), 2)} | "
              f"{f(mean([r['vram_peak_mib'] for r in rs]), 0)} |")


def boundary_table(recs):
    groups = defaultdict(list)
    for r in recs:
        groups[r["block"]].append(r)
    print("\n### capacity boundary\n")
    print("| block | max_context | kv_capacity | started | peak VRAM | request tokens |")
    print("|---|---:|---:|---|---:|---:|")
    for block in sorted(groups):
        rs = groups[block]
        cfg = rs[0]["config"]
        m = [r["metrics"] for r in rs]
        pt = f(mean([x["prompt_tokens"] for x in m]), 0)
        print(f"| {block} | {cfg.get('max_context')} | {cfg.get('kv_capacity')} | yes | "
              f"{f(mean([r['vram_peak_mib'] for r in rs]), 0)} | {pt} |")


def mtpk_table(recs):
    groups = defaultdict(list)
    for r in recs:
        groups[(r["block"], r["fixture"])].append(r)
    print("\n### MTP window sweep\n")
    print("| block | fixture | n | completion tok | decode tok/s | MTP accept | tokens/round | peak VRAM |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for k in sorted(groups):
        rs = groups[k]
        m = [r["metrics"] for r in rs]
        print(f"| {k[0]} | {k[1]} | {len(rs)} | {f(mean([x['completion_tokens'] for x in m]),0)} | "
              f"{f(mean([x['decode_tok_s'] for x in m]))} | {f(mean([x['spec_acceptance'] for x in m]),3)} | "
              f"{f(mean([x['spec_tokens_per_round'] for x in m]),2)} | {f(mean([r['vram_peak_mib'] for r in rs]),0)} |")


def main():
    ctx = load("ctx-sweep-greedy")
    for b in ("mtp3", "mtp0"):
        ctx_table(ctx, b)
    workload_table(load("workloads"))
    sweep_table(load("sweep"))
    mtpk_table(load("mtpk"))
    boundary_table(load("boundary"))
    stress = load("stress")
    print("\n### stress VRAM (per block, request order)\n")
    for block in sorted({r["block"] for r in stress}):
        vals = [r["vram_peak_mib"] for r in stress if r["block"] == block]
        print(f"- `{block}`: n={len(vals)} VRAM peaks {vals}")


if __name__ == "__main__":
    main()
