#!/bin/bash
# Sequentially run remaining campaigns after the context sweep finishes.
cd /validate || exit 1
while pgrep -f "run_campaign.py campaign_ctx.json" >/dev/null; do sleep 15; done
echo "[$(date +%T)] ctx campaign finished; starting remaining campaigns"
for c in workloads sweep stress boundary; do
  echo "[$(date +%T)] === campaign $c start ==="
  python3 -u run_campaign.py "campaign_$c.json" 2>&1 | tee "logs/$c.log"
  echo "[$(date +%T)] === campaign $c done ==="
done
echo "[$(date +%T)] ALL_DONE"
