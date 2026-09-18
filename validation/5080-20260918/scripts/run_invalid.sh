#!/bin/bash
# Start a short, vision-enabled server and run boundary/invalid-input probes.
set -uo pipefail
cd /validate || exit 1
mkdir -p out/invalid logs
port=18199
ctr=ninfer-invalid
docker rm -f "$ctr" >/dev/null 2>&1
docker run -d --name "$ctr" --gpus all --network host \
  -v /models:/ninfer-models:ro -v /repo:/repo:ro \
  -v /validate:/out -w /repo \
  --entrypoint /build/apps/ninfer-serve ninfer-mtpq4:build \
  /ninfer-models/qwen3_8_27b_minq4_mtpq4_visionq4.ninfer \
  --host 127.0.0.1 --port "$port" --model-id qwen3.8-27b-16gb \
  --max-context 65536 --kv-capacity 65536 --kv-dtype i4 \
  --spec mtp --draft-tokens 3 --prefill-chunk 64 --no-cuda-graph \
  --max-concurrency 2 --default-max-tokens 2048 --pending-timeout-ms 900000 \
  --no-prefix-reuse --vision --greedy --log-stats-interval-ms 0
ok=0
for i in $(seq 1 150); do curl -sf -m 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { ok=1; break; }; sleep 1; done
if [ "$ok" != "1" ]; then echo SERVER_NOT_READY; docker logs --tail 30 "$ctr"; exit 1; fi
python3 invalid.py --port "$port" --over-context /fixtures/niah_98304.json \
  --out out/invalid/results.json 2>&1 | tee logs/invalid.log
docker rm -f "$ctr" >/dev/null 2>&1
echo "[$(date +%T)] INVALID_DONE"
