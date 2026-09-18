#!/usr/bin/env python3
"""Aggregate run_campaign records.jsonl into summary JSON/CSV and run checks."""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = ["prompt_tokens", "completion_tokens", "prefill_seconds", "decode_seconds",
           "prefill_tok_s", "server_ttft_ms", "decode_tok_s", "spec_acceptance",
           "spec_tokens_per_round", "vram_peak_mib", "wall_seconds"]


def load(paths):
    recs = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if "metrics" in r:
                    recs.append(r)
    return recs


def group_key(r):
    c = r["config"]
    return (r["campaign"], r["block"], c.get("spec"), c.get("draft_tokens"), c.get("kv_dtype"),
            c.get("prefill_chunk"), c.get("cuda_graph"), c.get("vision"), c.get("lm_head_draft"),
            c.get("max_context"), r["fixture"])


def summarize(recs):
    groups = defaultdict(list)
    for r in recs:
        groups[group_key(r)].append(r)
    out = []
    for k, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        row = dict(zip(["campaign", "block", "spec", "draft_tokens", "kv_dtype", "prefill_chunk",
                        "cuda_graph", "vision", "lm_head_draft", "max_context", "fixture"], k))
        row["n"] = len(rs)
        for m in METRICS:
            vals = []
            for r in rs:
                v = r.get("metrics", {}).get(m, r.get(m))
                if v is not None:
                    try:
                        vals.append(float(v))
                    except (TypeError, ValueError):
                        pass
            if vals:
                row[m + "_mean"] = statistics.mean(vals)
                row[m + "_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                row[m + "_min"] = min(vals)
                row[m + "_max"] = max(vals)
        out.append(row)
    return out


def exactness(recs):
    """greedy MTP3 vs MTP0 identical-output check per (fixture, seed)."""
    by = {}
    for r in recs:
        if not r.get("greedy"):
            continue
        spec = r["config"].get("spec") or "none"
        by[(r["fixture"], r["seed"], spec, r["config"].get("max_context"))] = r
    results = []
    keys = {(f, s) for (f, s, _, _) in by}
    for f, s in sorted(keys, key=str):
        a = None
        b = None
        for (ff, ss, spec, mc), r in by.items():
            if (ff, ss) != (f, s):
                continue
            if spec == "mtp":
                a = r
            elif spec == "none":
                b = r
        if a and b:
            results.append({"fixture": f, "seed": s, "match": a["content_sha256"] == b["content_sha256"],
                            "mtp3_sha": a["content_sha256"], "mtp0_sha": b["content_sha256"],
                            "mtp3_chars": a["content_chars"], "mtp0_chars": b["content_chars"],
                            "mtp3_finish": a["metrics"]["finish_reason"], "mtp0_finish": b["metrics"]["finish_reason"]})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    recs = load(args.records)
    summary = summarize(recs)
    ex = exactness(recs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"records": len(recs), "summary": summary,
                                                  "spec_exactness": ex}, ensure_ascii=False, indent=1))
    fields = sorted({k for row in summary for k in row})
    with (out / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in summary:
            w.writerow(row)
    print(f"records={len(recs)} groups={len(summary)} exactness_pairs={len(ex)}")
    bad = [e for e in ex if not e["match"]]
    print(f"exactness mismatches={len(bad)}")
    for e in bad[:20]:
        print("  MISMATCH", e["fixture"], e["seed"])


if __name__ == "__main__":
    main()
