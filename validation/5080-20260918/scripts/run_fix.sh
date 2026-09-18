#!/bin/bash
cd /validate || exit 1
docker rm -f ninfer-bench >/dev/null 2>&1
for c in stress vision mtpk repeat; do
  echo "[$(date +%T)] === $c ==="
  python3 -u run_campaign.py "campaign_$c.json" 2>&1 | tee "logs/$c.log"
done
echo "[$(date +%T)] === conc ==="; bash run_conc.sh
echo "[$(date +%T)] === invalid ==="; bash run_invalid.sh
echo "[$(date +%T)] FIX_DONE"
