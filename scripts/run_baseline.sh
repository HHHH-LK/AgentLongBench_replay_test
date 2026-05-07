#!/usr/bin/env bash
# Vanilla 多轮 KV-reuse 基线 — 一键脚本
#
# 用法:
#   ./scripts/run_baseline.sh <dataset.jsonl> [output_dir] [limit]
#
# 环境变量:
#   BASE_URL   vLLM serve 地址, 默认 http://127.0.0.1:17100/v1
#   MODEL      模型名 (与 --served-model-name 一致), 默认 qwen3-30b-thinking
#   API_KEY    可选, 默认 EMPTY (本地服务不校验)

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "用法: $0 <dataset.jsonl> [output_dir] [limit]"
  exit 1
fi

DATASET=$1
OUTDIR=${2:-"runs/baseline_$(date +%Y%m%d_%H%M%S)"}
LIMIT=${3:-""}

BASE_URL=${BASE_URL:-"http://127.0.0.1:17100/v1"}
MODEL=${MODEL:-"qwen3-30b-thinking"}
API_KEY=${API_KEY:-"EMPTY"}

EXTRA=()
[[ -n "$LIMIT" ]] && EXTRA+=(--limit "$LIMIT")

echo "[bench] dataset=$DATASET"
echo "[bench] outdir =$OUTDIR"
echo "[bench] model  =$MODEL @ $BASE_URL"

python -m memory_bench.runner \
  --dataset "$DATASET" \
  --output-dir "$OUTDIR" \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --api-key "$API_KEY" \
  "${EXTRA[@]}"

python -m memory_bench.analyze --run-dir "$OUTDIR"

echo ""
echo "===== turn_summary.csv ====="
cat "$OUTDIR/turn_summary.csv"
echo ""
echo "===== bucket_summary.csv ====="
cat "$OUTDIR/bucket_summary.csv"
echo ""
echo "全部产物在: $OUTDIR"
