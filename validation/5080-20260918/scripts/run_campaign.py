#!/usr/bin/env python3
"""Fork-specific serving benchmark driver for NInfer on 16 GB Blackwell.

Metrics mirror upstream tools/bench/run_serve_corpus.py:
  prefill_tok_s = prompt_tokens / prefill_seconds
  server_ttft_ms = 1000 * (prepare + vision + prefill)
  decode_tok_s = (completion_tokens - 1) / decode_seconds
  spec_acceptance = accepted / drafted
  spec_tokens_per_round = 1 + accepted / rounds

It launches a dedicated docker container per configuration block, waits for
/health, warms up, then sends serial requests and reads exact phase timings from
the server's --request-log-jsonl. Raw records are appended to records.jsonl.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import subprocess
import threading
import time
from pathlib import Path

IMAGE_DEFAULT = "ninfer-mtpq4:build"
CONTAINER_DEFAULT = "ninfer-bench"


class CampaignError(RuntimeError):
    pass


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


class VramSampler:
    def __init__(self, interval: float = 0.25) -> None:
        self.interval = interval
        self.samples: list[tuple[float, int]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip().splitlines()
                if out:
                    self.samples.append((time.monotonic(), int(out[0].strip())))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def peak_between(self, t0: float, t1: float) -> int | None:
        vals = [v for (t, v) in self.samples if t0 <= t <= t1]
        return max(vals) if vals else None

    def current(self) -> int | None:
        return self.samples[-1][1] if self.samples else None


class ServerLog:
    """Incrementally reads the server JSONL and lets us await events."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.buffer = b""
        self.events: list[dict] = []

    def poll(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open("rb") as fh:
            fh.seek(self.offset)
            chunk = fh.read()
            self.offset = fh.tell()
        if not chunk:
            return []
        self.buffer += chunk
        lines = self.buffer.split(b"\n")
        self.buffer = lines.pop()
        new = []
        for raw in lines:
            if not raw.strip():
                continue
            ev = json.loads(raw)
            self.events.append(ev)
            new.append(ev)
        return new

    def await_event(self, name: str, timeout: float, after: float | None = None) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            for ev in self.poll():
                if ev.get("event") == name and (after is None or time.monotonic() >= after):
                    return ev
            if time.monotonic() >= deadline:
                raise CampaignError(f"timeout waiting for log event {name} in {self.path}")
            time.sleep(0.05)

    def take(self, name: str, start_index: int) -> dict | None:
        for i in range(start_index, len(self.events)):
            if self.events[i].get("event") == name:
                return self.events[i]
        return None


def docker_container_artifact(artifact: str) -> str:
    return artifact if artifact.startswith("/") else f"/ninfer-models/{artifact}"


def build_server_cmd(cfg: dict, campaign: dict, log_path_in_container: str) -> list[str]:
    cmd = [
        "docker", "run", "-d", "--name", campaign.get("container", CONTAINER_DEFAULT),
        "--gpus", "all", "--network", "host",
        "-w", "/repo",
        "-v", f"{campaign['models_dir']}:/ninfer-models:ro",
        "-v", f"{campaign['repo']}:/repo:ro",
        "-v", f"{campaign['out_dir']}:/out",
        "--entrypoint", "/build/apps/ninfer-serve",
        campaign.get("image", IMAGE_DEFAULT),
        docker_container_artifact(campaign["artifact"]),
        "--host", "127.0.0.1", "--port", str(campaign.get("port", 18199)),
        "--model-id", campaign.get("model_id", "qwen3.8-27b-16gb"),
        "--max-context", str(cfg["max_context"]),
        "--kv-capacity", str(cfg.get("kv_capacity", cfg["max_context"])),
        "--kv-dtype", cfg.get("kv_dtype", "i4"),
        "--prefill-chunk", str(cfg.get("prefill_chunk", 64)),
        "--max-concurrency", str(cfg.get("max_concurrency", 1)),
        "--default-max-tokens", str(cfg.get("default_max_tokens", 4096)),
        "--pending-timeout-ms", "900000",
        "--log-stats-interval-ms", "0",
        "--request-log-jsonl", log_path_in_container,
    ]
    if cfg.get("spec"):
        cmd += ["--spec", cfg["spec"], "--draft-tokens", str(cfg.get("draft_tokens", 3))]
    if cfg.get("vision"):
        cmd.append("--vision")
    if cfg.get("lm_head_draft"):
        cmd.append("--lm-head-draft")
    if not cfg.get("cuda_graph", False):
        cmd.append("--no-cuda-graph")
    if not cfg.get("prefix_reuse", False):
        cmd.append("--no-prefix-reuse")
    if cfg.get("greedy"):
        cmd.append("--greedy")
    elif cfg.get("stochastic_profile", "upstream") == "upstream":
        # Match upstream docs/performance.md stochastic profile.
        cmd += ["--temperature", "0.6", "--top-p", "0.95", "--top-k", "20",
                "--min-p", "0", "--presence-penalty", "1.0", "--frequency-penalty", "0"]
    if cfg.get("no_thinking"):
        cmd.append("--no-thinking")
    if cfg.get("preserve_thinking"):
        cmd.append("--preserve-thinking")
    return cmd


def wait_health(port: int, container: str, timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        poll = run(["docker", "inspect", "-f", "{{.State.Running}}", container])
        if poll.stdout.strip() != "true":
            logs = run(["docker", "logs", "--tail", "40", container])
            raise CampaignError(f"container stopped during startup:\n{logs.stdout}\n{logs.stderr}")
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            body = resp.read()
            conn.close()
            if resp.status == 200 and json.loads(body).get("status") == "ok":
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise CampaignError("timeout waiting for /health")


def load_fixture(repo: Path, name: str, fixtures_dir: Path | None = None) -> dict:
    if fixtures_dir is not None and (fixtures_dir / f"{name}.json").exists():
        msgs = json.loads((fixtures_dir / f"{name}.json").read_text())
        if isinstance(msgs, dict):
            msgs = msgs["messages"]
        return {"case": {"name": name}, "messages": msgs, "thinking": False, "max_new": 128}
    manifest = json.loads((repo / "examples/cli/manifest.json").read_text())
    cases = {c["name"]: c for c in manifest["cases"]}
    case = cases[name]
    msgs = json.loads((repo / "examples/cli" / case["messages"]).read_text())
    if isinstance(msgs, dict):
        msgs = msgs["messages"]
    return {"case": case, "messages": msgs, "thinking": bool(case["thinking"]), "max_new": int(case["max_new"])}


def post_request(port: int, model_id: str, messages: list, max_new: int, thinking: bool, seed: int, timeout: float) -> tuple[dict, float]:
    body = json.dumps({
        "model": model_id,
        "messages": messages,
        "max_completion_tokens": max_new,
        "seed": seed,
        "stream": False,
        "enable_thinking": thinking,
    }, ensure_ascii=False).encode("utf-8")
    t0 = time.monotonic()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("POST", "/v1/chat/completions", body=body, headers={
            "Content-Type": "application/json", "Content-Length": str(len(body)), "Connection": "keep-alive"})
        resp = conn.getresponse()
        raw = resp.read()
    finally:
        conn.close()
    wall = time.monotonic() - t0
    if resp.status != 200:
        raise CampaignError(f"HTTP {resp.status}: {raw[:500].decode('utf-8', 'replace')}")
    return json.loads(raw), wall


def extract_metrics(event: dict) -> dict:
    r = event.get("result", {})
    t = event.get("timings_seconds", {})
    s = event.get("speculative", {})
    prompt = int(r.get("prompt_tokens", 0))
    completion = int(r.get("completion_tokens", 0))
    prefill = float(t.get("prefill", 0.0))
    decode = float(t.get("decode", 0.0))
    prepare = float(t.get("prepare", 0.0))
    vision = float(t.get("vision", 0.0))
    total = float(t.get("total", 0.0))
    rounds = int(s.get("rounds", 0))
    drafted = int(s.get("drafted_tokens", 0))
    accepted = int(s.get("accepted_tokens", 0))
    return {
        "prompt_tokens": prompt,
        "computed_prefill_tokens": int(r.get("computed_prefill_tokens", 0)),
        "completion_tokens": completion,
        "finish_reason": r.get("finish_reason"),
        "prefix_cache_hit_tokens": int(r.get("prefix_cache_hit_tokens", 0)),
        "prepare_seconds": prepare,
        "vision_seconds": vision,
        "prefill_seconds": prefill,
        "decode_seconds": decode,
        "total_seconds": total,
        "ttft_seconds": float(t.get("ttft", prepare + vision + prefill)),
        "prefill_tok_s": prompt / prefill if prefill > 0 else None,
        "server_ttft_ms": 1000.0 * (prepare + vision + prefill),
        "decode_tok_s": (completion - 1) / decode if decode > 0 and completion > 0 else None,
        "spec_backend": s.get("backend"),
        "spec_draft_window": s.get("draft_window"),
        "spec_rounds": rounds,
        "spec_drafted": drafted,
        "spec_accepted": accepted,
        "spec_acceptance": accepted / drafted if drafted > 0 else None,
        "spec_tokens_per_round": (1.0 + accepted / rounds) if rounds > 0 else None,
        "spec_fallback_steps": int(s.get("fallback_steps", 0)),
        "accepted_per_position": s.get("accepted_per_position"),
    }


def run_block(campaign: dict, block: dict, out_dir: Path, sampler: VramSampler, records_fh) -> None:
    tag = block["tag"]
    cfg = dict(campaign.get("default_config", {}))
    cfg.update(block.get("config", {}))
    cfg.setdefault("max_context", 124928)
    cfg.setdefault("kv_capacity", cfg["max_context"])
    container = campaign.get("container", CONTAINER_DEFAULT)
    port = campaign.get("port", 18199)
    repo = Path(campaign["repo"])
    fixtures_dir = Path(campaign["fixtures_dir"]) if campaign.get("fixtures_dir") else None
    run(["docker", "rm", "-f", container])
    server_dir = out_dir / "server"
    server_dir.mkdir(parents=True, exist_ok=True)
    log_host = server_dir / f"{tag}.jsonl"
    log_host.unlink(missing_ok=True)
    log_container = f"/out/server/{tag}.jsonl"
    cmd = build_server_cmd(cfg, campaign, log_container)
    print(f"[{time.strftime('%T')}] block {tag}: starting container")
    started = run(cmd)
    if started.returncode != 0:
        raise CampaignError(f"docker run failed: {started.stderr}")
    try:
        wait_health(port, container)
        log = ServerLog(log_host)
        start_event = log.await_event("server_start", timeout=180.0)
        engine = start_event.get("engine", {})
        load_seconds = start_event.get("artifact", {}).get("load_seconds")
        server_instance = start_event.get("server_instance_id")
        print(f"[{time.strftime('%T')}] block {tag}: ready load={load_seconds}s engine={json.dumps(engine, ensure_ascii=False)}")
        (out_dir / f"server_start_{tag}.json").write_text(json.dumps(start_event, ensure_ascii=False, indent=1))
        vram_after_load = sampler.current()

        # Warm-up (not measured)
        warm_name = block.get("warmup_fixture", "text_smoke_zh")
        warm = load_fixture(repo, warm_name, fixtures_dir)
        for _ in range(int(block.get("warmup_rounds", 2))):
            try:
                post_request(port, campaign["model_id"], warm["messages"], warm["max_new"],
                             warm["thinking"], 1234, timeout=campaign.get("request_timeout", 3600))
            except Exception as exc:  # noqa: BLE001
                print(f"[{time.strftime('%T')}] block {tag}: warmup warning {exc}")
            log.poll()

        pending = list(block["requests"])
        for idx, req in enumerate(pending):
            name = req["fixture"]
            fixture = load_fixture(repo, name, fixtures_dir)
            max_new = int(req.get("max_new", fixture["max_new"]))
            thinking = bool(req.get("thinking", fixture["thinking"]))
            seed = int(req.get("seed", 42 + idx))
            events_before = len(log.events)
            t0 = time.monotonic()
            response, wall = post_request(port, campaign["model_id"], fixture["messages"], max_new,
                                          thinking, seed, timeout=campaign.get("request_timeout", 3600))
            t1 = time.monotonic()
            log.poll()
            done = log.take("request_done", events_before)
            if done is None:
                log.poll()
                done = log.take("request_done", events_before)
            if done is None:
                raise CampaignError(f"no request_done event for {tag} {name} seed {seed}")
            metrics = extract_metrics(done)
            message = response["choices"][0]["message"]
            content = message.get("content") or ""
            reasoning = message.get("reasoning_content") or ""
            vram_peak = sampler.peak_between(t0, t1)
            content_dir = out_dir / "content"
            record = {
                "campaign": campaign.get("name"),
                "block": tag,
                "config": cfg,
                "fixture": name,
                "category": req.get("category"),
                "seed": seed,
                "max_new": max_new,
                "thinking": thinking,
                "greedy": bool(cfg.get("greedy")),
                "wall_seconds": wall,
                "vram_peak_mib": vram_peak,
                "vram_after_load_mib": vram_after_load,
                "load_seconds": load_seconds,
                "server_instance_id": server_instance,
                "content_sha256": sha256_text(content),
                "content_chars": len(content),
                "reasoning_chars": len(reasoning),
                "content_head": content[:200],
                "content_tail": content[-200:],
                "finish_reason_http": response["choices"][0].get("finish_reason"),
                "metrics": metrics,
                "ts": time.time(),
            }
            if content or reasoning:
                content_dir.mkdir(exist_ok=True)
                cname = f"{tag}__{name}__{seed}.txt"
                blob = content
                if reasoning:
                    blob += "\n<<<REASONING>>>\n" + reasoning
                if len(blob) > 2_000_000:
                    blob = blob[:2_000_000] + "\n<<<TRUNCATED>>>"
                (content_dir / cname).write_text(blob)
                record["content_file"] = str(content_dir / cname)
            records_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            records_fh.flush()
            os.fsync(records_fh.fileno())
            print(f"[{time.strftime('%T')}] {tag} {name} seed={seed} pt={metrics['prompt_tokens']} "
                  f"ct={metrics['completion_tokens']} prefill={metrics['prefill_tok_s'] and round(metrics['prefill_tok_s'],1)} "
                  f"decode={metrics['decode_tok_s'] and round(metrics['decode_tok_s'],1)} "
                  f"acc={metrics['spec_acceptance'] and round(metrics['spec_acceptance'],3)} vram={vram_peak}")
    finally:
        logs = run(["docker", "logs", "--tail", "30", container])
        (out_dir / f"docker_{tag}.log").write_text(logs.stdout + logs.stderr)
        run(["docker", "rm", "-f", container])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign")
    ap.add_argument("--only", action="append", default=None, help="run only these block tags")
    args = ap.parse_args()
    campaign = json.loads(Path(args.campaign).read_text())
    out_dir = Path(campaign["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    sampler = VramSampler()
    sampler.start()
    records_path = out_dir / "records.jsonl"
    with records_path.open("a", encoding="utf-8") as fh:
        for block in campaign["blocks"]:
            if args.only and block["tag"] not in args.only:
                continue
            try:
                run_block(campaign, block, out_dir, sampler, fh)
            except Exception as exc:  # noqa: BLE001
                print(f"[{time.strftime('%T')}] block {block['tag']} FAILED: {exc}")
                fh.write(json.dumps({"campaign": campaign.get("name"), "block": block["tag"],
                                     "config": dict(campaign.get("default_config", {}), **block.get("config", {})),
                                     "failure": str(exc), "ts": time.time()}, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
                run(["docker", "rm", "-f", campaign.get("container", CONTAINER_DEFAULT)])
    sampler.stop()
    print(f"done -> {records_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
