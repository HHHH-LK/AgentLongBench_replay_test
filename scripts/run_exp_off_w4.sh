#!/usr/bin/env bash
# 跑 LMCache OFF, workers=4 一组 (前提: 远程 vLLM 已切到无 LMCache)
set -euo pipefail
cd "$(dirname "$0")/.."

: "${BASE_URL:=http://10.26.6.54:17100/v1}"
: "${MODEL:=qwen3-30b-thinking}"
: "${API_KEY:=EMPTY}"
: "${DATASET:=/Users/lk_hhh/Documents/TAOmem1/benchmarks/datasets/agentlongbench/benchmark/ki-v/64k/final_guess/intersection.jsonl}"
: "${LIMIT:=12}"
: "${MAX_TOKENS:=2048}"
: "${TEMPERATURE:=0.6}"

log() { echo -e "\n\033[1;36m[$(date +%H:%M:%S)] $*\033[0m"; }
ok()  { echo -e "\033[0;32m  ✓ $*\033[0m"; }
err() { echo -e "\033[0;31m  ✗ $*\033[0m" >&2; }

log "验证 vLLM 可达 ($BASE_URL)"
code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$BASE_URL/models" 2>/dev/null || echo 000)
[[ "$code" == "200" ]] || { err "服务不可达 (http_code=$code)"; exit 1; }
ok "服务就绪"

mkdir -p runs
out=runs/exp_off_w4

log "═══ 实验 [exp_off_w4] workers=4 limit=$LIMIT ═══"
uv run python -m memory_bench.runner \
    --dataset "$DATASET" \
    --output-dir "$out" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --api-key "$API_KEY" \
    --temperature "$TEMPERATURE" \
    --max-tokens "$MAX_TOKENS" \
    --limit "$LIMIT" \
    --workers 4 \
    --run-id exp_off_w4 \
    --disable-thinking 2>&1 | tee "$out.runner.log"

uv run python -m memory_bench.analyze --run-dir "$out" 2>&1 | tee "$out.analyze.log"
ok "[exp_off_w4] 完成 → $out"

echo ""
echo "━━━━━━━━ 三组对比 ━━━━━━━━"
for d in runs/exp_on_w1 runs/exp_off_w4 runs/exp_on_w4; do
    [[ -d "$d" ]] || continue
    name=$(basename "$d")
    echo ""
    echo "=== $name ==="
    [[ -f "$d/turn_summary.csv" ]] && cat "$d/turn_summary.csv"
    if [[ -f "$d/accuracy.csv" ]]; then
        avg=$(awk -F, 'NR>1 && $7!="" && $7!="None" {s+=$7; n++} END {if(n>0) printf "%.4f", s/n; else print "N/A"}' "$d/accuracy.csv")
        echo "[accuracy] avg=$avg"
    fi
done
