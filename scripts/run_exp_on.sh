#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export BASE_URL=http://10.26.6.54:17100/v1
export MODEL=qwen3-30b-thinking
DATASET=/Users/lk_hhh/Documents/TAOmem/benchmarks/datasets/agentlongbench/benchmark/ki-v/64k/final_guess/intersection.jsonl

echo ">>> 实验 1: LMCache ON, workers=1"
uv run python -m memory_bench.runner \
  --dataset "$DATASET" \
  --output-dir runs/exp_on_w1 \
  --base-url "$BASE_URL" --model "$MODEL" \
  --max-tokens 2048 --temperature 0.6 \
  --disable-thinking --limit 12 --workers 1 \
  --run-id on_w1 2>&1 | tee runs/exp_on_w1.log

uv run python -m memory_bench.analyze --run-dir runs/exp_on_w1

echo ">>> 休息 30 秒"
sleep 30

echo ">>> 实验 3: LMCache ON, workers=4"
uv run python -m memory_bench.runner \
  --dataset "$DATASET" \
  --output-dir runs/exp_on_w4 \
  --base-url "$BASE_URL" --model "$MODEL" \
  --max-tokens 2048 --temperature 0.6 \
  --disable-thinking --limit 12 --workers 4 \
  --run-id on_w4 2>&1 | tee runs/exp_on_w4.log

uv run python -m memory_bench.analyze --run-dir runs/exp_on_w4

echo ">>> 完成。结果在 runs/exp_on_w1 和 runs/exp_on_w4"
