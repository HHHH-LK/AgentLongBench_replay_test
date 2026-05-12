#!/usr/bin/env bash
# 跑两组 LMCache ON 实验 (workers=1, workers=4), 同一个 vLLM 不重启
#
# 实验设计:
#   exp_on_w1   workers=1   (单并发, 纯 KV 复用)
#   exp_on_w4   workers=4   (高并发 + KV 复用)
#
# 用完之后再去远程切 vLLM 到 LMCache OFF, 跑 exp_off_w4

set -euo pipefail

cd "$(dirname "$0")/.."

# ─── 配置 ───────────────────────────────────────────
: "${BASE_URL:=http://10.26.6.54:17100/v1}"
: "${MODEL:=qwen3-30b-thinking}"
: "${API_KEY:=EMPTY}"
: "${DATASET:=/Users/lk_hhh/Documents/TAOmem1/benchmarks/datasets/agentlongbench/benchmark/ki-v/64k/final_guess/intersection.jsonl}"
: "${LIMIT:=12}"
: "${MAX_TOKENS:=2048}"
: "${TEMPERATURE:=0.6}"
: "${SLEEP_BETWEEN:=30}"

mkdir -p runs

log() { echo -e "\n\033[1;36m[$(date +%H:%M:%S)] $*\033[0m"; }
ok()  { echo -e "\033[0;32m  ✓ $*\033[0m"; }
err() { echo -e "\033[0;31m  ✗ $*\033[0m" >&2; }

# ─── 验证服务 ────────────────────────────────────────
log "验证 vLLM 可达 ($BASE_URL)"
code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$BASE_URL/models" 2>/dev/null || echo 000)
if [[ "$code" != "200" ]]; then
    err "服务不可达 (http_code=$code)"
    exit 1
fi
ok "服务就绪"

# ─── 单组跑 ─────────────────────────────────────────
run_one() {
    local name=$1
    local workers=$2
    local out="runs/$name"

    log "═══ 实验 [$name] workers=$workers limit=$LIMIT ═══"

    if uv run python -m memory_bench.runner \
        --dataset "$DATASET" \
        --output-dir "$out" \
        --base-url "$BASE_URL" \
        --model "$MODEL" \
        --api-key "$API_KEY" \
        --temperature "$TEMPERATURE" \
        --max-tokens "$MAX_TOKENS" \
        --limit "$LIMIT" \
        --workers "$workers" \
        --run-id "$name" \
        --disable-thinking 2>&1 | tee "$out.runner.log"; then
        ok "[$name] runner 完成"
    else
        err "[$name] runner 失败"
        return 1
    fi

    uv run python -m memory_bench.analyze --run-dir "$out" 2>&1 | tee "$out.analyze.log"
    ok "[$name] 全部完成 → $out"
}

# ─── 主流程 ─────────────────────────────────────────
log "≡≡≡ 两组 LMCache ON 对比 (w=1 / w=4) ≡≡≡"

run_one "exp_on_w1" 1

log "组间休息 ${SLEEP_BETWEEN} 秒..."
sleep "$SLEEP_BETWEEN"

run_one "exp_on_w4" 4

# ─── 汇总 ───────────────────────────────────────────
log "≡≡≡ 全部完成 ≡≡≡"
echo ""
for d in runs/exp_on_w1 runs/exp_on_w4; do
    name=$(basename "$d")
    echo "━━━━━━━━━━━━━━━━━━━━━━━━ $name ━━━━━━━━━━━━━━━━━━━━━━━━"
    if [[ -f "$d/turn_summary.csv" ]]; then
        cat "$d/turn_summary.csv"
    else
        echo "(no turn_summary.csv)"
    fi
    if [[ -f "$d/accuracy.csv" ]]; then
        echo ""
        echo "[accuracy]"
        n_total=$(($(wc -l < "$d/accuracy.csv") - 1))
        n_scored=$(awk -F, 'NR>1 && $7!="" && $7!="None" {n++} END {print n+0}' "$d/accuracy.csv")
        avg=$(awk -F, 'NR>1 && $7!="" && $7!="None" {s+=$7; n++} END {if(n>0) printf "%.4f", s/n; else print "N/A"}' "$d/accuracy.csv")
        echo "n_total=$n_total  n_scored=$n_scored  avg_score=$avg"
    fi
    echo ""
done

cat <<EOF

下一步: 切 vLLM 到 LMCache OFF, 然后跑 exp_off_w4
  ./scripts/run_exp_off_w4.sh
EOF
