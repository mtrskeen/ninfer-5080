#!/bin/bash
set -uo pipefail
cd /validate || exit 1
mkdir -p out/lmeval logs
port=18199; ctr=ninfer-eval
docker rm -f "$ctr" >/dev/null 2>&1
docker run -d --name "$ctr" --gpus all --network host \
  -v /models:/ninfer-models:ro -v /validate:/out \
  --entrypoint /build/apps/ninfer-serve ninfer-mtpq4:build \
  /ninfer-models/qwen3_8_27b_minq4_mtpq4_visionq4.ninfer \
  --host 127.0.0.1 --port "$port" --model-id qwen3.8-27b-16gb \
  --max-context 124928 --kv-capacity 124928 --kv-dtype i4 \
  --spec mtp --draft-tokens 3 --prefill-chunk 256 --no-cuda-graph \
  --max-concurrency 1 --default-max-tokens 2048 --pending-timeout-ms 900000 \
  --no-prefix-reuse --no-thinking --greedy --log-stats-interval-ms 0
for i in $(seq 1 150); do curl -sf -m 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break; sleep 1; done
export OPENAI_API_KEY=dummy
EV=/evalenv/bin/lm-eval
MA="model=qwen3.8-27b-16gb,base_url=http://127.0.0.1:$port/v1/chat/completions,tokenizer=/validate/qwen38_tokenizer,num_concurrent=1,max_retries=2,timeout=3600"
$EV --model openai-chat-completions --model_args "$MA" \
  --tasks niah_single_1 \
  --metadata '{"max_seq_lengths":[8192,32768,65536]}' \
  --limit 3 --apply_chat_template --confirm_run_unsafe_code \
  --output_path /out/lmeval_ruler2 2>&1 | tee logs/ruler2.log
echo "[$(date +%T)] RULER2_DONE"
docker rm -f "$ctr" >/dev/null 2>&1
