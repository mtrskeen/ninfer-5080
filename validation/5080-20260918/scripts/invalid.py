#!/usr/bin/env python3
"""Send malformed / boundary requests to a running ninfer-serve and verify the
server survives and rejects them cleanly."""
import argparse
import http.client
import json
import time
from pathlib import Path

CASES_BASIC = [
    ("empty_messages", {"messages": [], "max_completion_tokens": 8}),
    ("missing_role", {"messages": [{"content": "hi"}], "max_completion_tokens": 8}),
    ("bad_role", {"messages": [{"role": "wizard", "content": "hi"}], "max_completion_tokens": 8}),
    ("unknown_model", {"model": "nope", "messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 8}),
    ("zero_max_tokens", {"messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 0}),
    ("negative_max_tokens", {"messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": -5}),
    ("huge_max_tokens", {"messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 1000000}),
    ("unknown_content_type", {"messages": [{"role": "user", "content": [{"type": "hologram", "data": "x"}]}], "max_completion_tokens": 8}),
    ("bad_image_path", {"messages": [{"role": "user", "content": [{"type": "image", "image": "examples/cli/media/does_not_exist.png"}, {"type": "text", "text": "what?"}]}], "max_completion_tokens": 8}),
    ("empty_content", {"messages": [{"role": "user", "content": ""}], "max_completion_tokens": 8}),
]
RAW_CASES = [
    ("malformed_json", b"{not json"),
]


def send_raw(port: int, model: str, payload: dict | None, raw: bytes | None, timeout: float = 120.0):
    body = raw if raw is not None else json.dumps(payload, ensure_ascii=False).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    status, text = None, ""
    try:
        conn.request("POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        text = resp.read().decode("utf-8", "replace")
        status = resp.status
    except Exception as exc:  # noqa: BLE001
        status, text = -1, f"{type(exc).__name__}: {exc}"
    finally:
        conn.close()
    return status, text[:400]


def health(port: int) -> dict:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("GET", "/health")
        resp = conn.getresponse()
        return {"status": resp.status, "body": resp.read().decode("utf-8", "replace")[:200]}
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18199)
    ap.add_argument("--model", default="qwen3.8-27b-16gb")
    ap.add_argument("--out", required=True)
    ap.add_argument("--over-context", help="fixture JSON that exceeds the context")
    args = ap.parse_args()
    results = []
    for name, payload in CASES_BASIC:
        payload = dict(payload)
        payload.setdefault("model", args.model)
        st, body = send_raw(args.port, args.model, payload, None)
        results.append({"case": name, "status": st, "body": body, "health": health(args.port)})
        print(name, st, body[:120].replace("\n", " "))
    for name, raw in RAW_CASES:
        st, body = send_raw(args.port, args.model, None, raw)
        results.append({"case": name, "status": st, "body": body, "health": health(args.port)})
        print(name, st, body[:120].replace("\n", " "))
    if args.over_context:
        msgs = json.loads(Path(args.over_context).read_text())
        if isinstance(msgs, dict):
            msgs = msgs["messages"]
        st, body = send_raw(args.port, args.model, {"model": args.model, "messages": msgs,
                                                    "max_completion_tokens": 4096}, None, timeout=900)
        results.append({"case": "over_context", "status": st, "body": body, "health": health(args.port)})
        print("over_context", st, body[:200].replace("\n", " "))
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print("health_after_all", health(args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
