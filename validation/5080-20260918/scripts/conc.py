#!/usr/bin/env python3
"""Concurrent serving test: C workers released together on one persistent server."""
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, "/validate")
from run_campaign import (CampaignError, ServerLog, VramSampler, build_server_cmd,  # noqa: E402
                          docker_container_artifact, extract_metrics, load_fixture,
                          post_request, run, wait_health)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18199)
    ap.add_argument("--container", default="ninfer-bench")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--kv-capacity", type=int, default=32768)
    ap.add_argument("--max-context", type=int, default=16384)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--fixture", default="decode_prose")
    ap.add_argument("--spec", default="mtp")
    ap.add_argument("--draft-tokens", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    repo = Path("/repo")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"max_context": args.max_context, "kv_capacity": args.kv_capacity, "kv_dtype": "i4",
           "prefill_chunk": 64, "cuda_graph": False, "prefix_reuse": False, "greedy": True,
           "max_concurrency": args.concurrency, "spec": args.spec or None,
           "draft_tokens": args.draft_tokens}
    campaign = {"repo": str(repo), "models_dir": "/models", "artifact": "qwen3_8_27b_minq4_mtpq4_visionq4.ninfer",
                "model_id": "qwen3.8-27b-16gb", "out_dir": str(out_dir.resolve()), "port": args.port,
                "container": args.container, "image": "ninfer-mtpq4:build"}
    log_host = out_dir / "server" / "conc.jsonl"
    log_host.parent.mkdir(exist_ok=True)
    log_host.unlink(missing_ok=True)
    run(["docker", "rm", "-f", args.container])
    cmd = build_server_cmd(cfg, campaign, "/out/server/conc.jsonl")
    print("starting:", " ".join(cmd))
    started = run(cmd)
    if started.returncode != 0:
        raise SystemExit(started.stderr)
    sampler = VramSampler()
    sampler.start()
    results = []
    try:
        wait_health(args.port, args.container)
        log = ServerLog(log_host)
        start = log.await_event("server_start", 180)
        server_instance = start["server_instance_id"]
        print("ready", json.dumps(start.get("engine"), ensure_ascii=False))
        fixture = load_fixture(repo, args.fixture, Path("/fixtures"))
        responses = [None] * args.concurrency
        t0 = time.monotonic()
        threads = []
        for i in range(args.concurrency):
            def worker(i=i):
                r, wall = post_request(args.port, "qwen3.8-27b-16gb", fixture["messages"], args.max_new,
                                       False, 7000 + i, timeout=3600)
                responses[i] = (r, wall)
            th = threading.Thread(target=worker)
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
        makespan = time.monotonic() - t0
        time.sleep(0.5)
        log.poll()
        dones = [e for e in log.events if e.get("event") == "request_done"]
        metrics = [extract_metrics(e) for e in dones]
        total_decode_tokens = sum(m["completion_tokens"] - 1 for m in metrics)
        total_drafted = sum(m["spec_drafted"] for m in metrics)
        total_accepted = sum(m["spec_accepted"] for m in metrics)
        record = {"concurrency": args.concurrency, "kv_capacity": args.kv_capacity, "max_context": args.max_context,
                  "max_new": args.max_new, "fixture": args.fixture, "makespan_seconds": makespan,
                  "per_request": metrics, "aggregate_decode_tok_s": total_decode_tokens / makespan,
                  "aggregate_acceptance": total_accepted / total_drafted if total_drafted else None,
                  "vram_peak_mib": sampler.peak_between(t0 - 30, time.monotonic())}
        results.append(record)
        print(json.dumps({k: record[k] for k in ("concurrency", "makespan_seconds", "aggregate_decode_tok_s", "aggregate_acceptance", "vram_peak_mib")}))
        (out_dir / "conc_result.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    finally:
        sampler.stop()
        run(["docker", "rm", "-f", args.container])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
