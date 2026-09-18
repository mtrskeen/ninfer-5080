#!/bin/bash
# Concurrent serving points (upstream decode-saturation style). Continues on failure.
cd /validate || exit 1
mkdir -p out/conc logs
run() {
  local C=$1 MC=$2 KV=$3
  echo "[$(date +%T)] conc C=$C max_context=$MC kv=$KV"
  python3 conc.py --concurrency "$C" --max-context "$MC" --kv-capacity "$KV" \
    --max-new 2048 --fixture decode_prose \
    --out "out/conc/C${C}-mc${MC}" >> logs/conc.log 2>&1
  echo "[$(date +%T)] rc=$? C=$C mc=$MC"
}
run 1 16384 16384
run 2 16384 32768
run 4 16384 65536
run 2 32768 65536
run 4 24576 98304
echo "[$(date +%T)] CONC_DONE"
