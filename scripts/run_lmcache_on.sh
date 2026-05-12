#!/usr/bin/env bash
# 单 vLLM 实例上跑 3 组实验 (假设服务已启动且 LMCache ON, 关 thinking)
#
# 实验设计:
#   01_w1   workers=1  (无并发基线, KV 复用最纯净)
#   02_w2   workers=2  (中度并发)
#   03_w4   workers=4  (高并发, 看吞吐 vs 延迟)
#
# 用法:
#   export DATASET=/path/to/intersection.jsonl
#   export LIMIT=20                  # 跑前 N 条
#   ./scripts/run_lmcache_on.sh
#
# 注意:
#   - 不重启 vLLM, 同一服务连跑三组 (中间也不主动清缓存)
#   - 关 thinking (chat_template_kwargs.enable_thinking=false)
#   - 每组之间会 sleep 30 秒, 避免上一组的请求未完全释放就开下一组

set -euo pipefail

# ────────────────────────────────────────────────────────────────
# 配置
# ────────────────────────────────────────────────────────────────
: "${DATASET:?需要 export DATASET=/path/to/dataset.jsonl}"
: "${BASE_URL:=http://10.26.6.54:17100/v1}"
: "${MODEL:=qwen3-30b-thinking}"
: "${API_KEY:=EMPTY}"

: "${LIMIT:=20}"                      # 默认跑 20 条
: "${MAX_TOKENS:=2048}"               # 关 thinking 后 2048 够用
: "${TEMPERATURE:=0.6}"

# 三组实验的 worker 数 (可改)
: "${W1:=1}"
: "${W2:=2}"
: "${W3:=4}"

: "${SLEEP_BETWEEN:=30}"              # 组间间隔 (秒)

ROOT=$(cd "$(dirname "$0")/.." && pwd)
TS=$(date +%Y%m%d_%H%M%S)
EXP_ROOT="$ROOT/runs/lmcache_on_${TS}"
mkdir -p "$EXP_ROOT"

# ────────────────────────────────────────────────────────────────
# 工具函数
# ────────────────────────────────────────────────────────────────
log()   { echo -e "\n\033[1;36m[$(date +%H:%M:%S)] $*\033[0m"; }
info()  { echo -e "\033[0;33m  → $*\033[0m"; }
ok()    { echo -e "\033[0;32m  ✓ $*\033[0m"; }
err()   { echo -e "\033[0;31m  ✗ $*\033[0m" >&2; }

verify_service() {
    log "验证 vLLM 服务可达 ($BASE_URL)"
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 \
           "$BASE_URL/models" 2>/dev/null || echo 000)
    if [[ "$code" != "200" ]]; then
        err "服务不可达 (http_code=$code), 请先确认 vLLM 已启动并 LMCache ON"
        exit 1
    fi
    ok "服务就绪"

    # 同时测一下 disable_thinking 是不是真生效
    info "测试 disable_thinking 是否生效..."
    local resp
    resp=$(curl -s --max-time 30 "$BASE_URL/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
          "model":"'$MODEL'",
          "messages":[{"role":"user","content":"What is 2+2?"}],
          "max_tokens":50,
          "chat_template_kwargs":{"enable_thinking":false}
        }' 2>/dev/null || echo "")
    if [[ -z "$resp" ]]; then
        err "测试请求失败"
        exit 1
    fi
    local content tokens
    content=$(echo "$resp" | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'][:80])" 2>/dev/null || echo "")
    tokens=$(echo "$resp" | python3 -c "import json,sys; print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null || echo "?")
    info "测试输出: $content"
    info "completion_tokens: $tokens"
    if [[ "$content" == *"<think>"* ]]; then
        err "输出含 <think> 段, disable_thinking 没生效"
        err "你的 vLLM 可能不支持 chat_template_kwargs, 请确认版本"
        exit 1
    fi
    ok "disable_thinking 生效"
}

run_phase() {
    local name=$1
    local workers=$2
    local out="$EXP_ROOT/$name"

    log "═══ 实验 [$name] workers=$workers ═══"
    info "数据集: $(basename "$DATASET")"
    info "样本数: $LIMIT"
    info "输出目录: $out"

    cd "$ROOT"
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
        err "[$name] runner 失败 (见 $out.runner.log)"
        return 1
    fi

    uv run python -m memory_bench.analyze --run-dir "$out" 2>&1 | tee "$out.analyze.log"
    ok "[$name] 完成 → $out"
}

# ────────────────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────────────────
log "≡≡≡ 三组并发对比实验 (LMCache ON, disable_thinking) ≡≡≡"
info "数据集: $DATASET"
info "样本数: $LIMIT (每组都跑这么多)"
info "max_tokens: $MAX_TOKENS  temperature: $TEMPERATURE"
info "workers 配置: $W1 / $W2 / $W3"
info "服务地址: $BASE_URL"
info "结果目录: $EXP_ROOT"

verify_service

# ─── 实验 1 ───
run_phase "01_w${W1}" "$W1"
log "组间休息 ${SLEEP_BETWEEN} 秒..."
sleep "$SLEEP_BETWEEN"

# ─── 实验 2 ───
run_phase "02_w${W2}" "$W2"
log "组间休息 ${SLEEP_BETWEEN} 秒..."
sleep "$SLEEP_BETWEEN"

# ─── 实验 3 ───
run_phase "03_w${W3}" "$W3"

# ────────────────────────────────────────────────────────────────
# 自动汇总
# ────────────────────────────────────────────────────────────────
log "≡≡≡ 全部完成 ≡≡≡"
echo ""
echo "结果目录: $EXP_ROOT"
echo ""

for d in "$EXP_ROOT"/*/; do
    [[ -d "$d" ]] || continue
    name=$(basename "$d")
    echo "━━━━━━━━━━━━━━━━━━━━━━━━ $name ━━━━━━━━━━━━━━━━━━━━━━━━"
    if [[ -f "$d/turn_summary.csv" ]]; then
        cat "$d/turn_summary.csv"
    else
        echo "(无 turn_summary.csv)"
    fi
    if [[ -f "$d/accuracy.csv" ]]; then
        echo ""
        echo "[accuracy]"
        # 简单统计
        n_total=$(($(wc -l < "$d/accuracy.csv") - 1))
        n_scored=$(awk -F, 'NR>1 && $7 != "" && $7 != "None" {n++} END {print n+0}' "$d/accuracy.csv")
        avg=$(awk -F, 'NR>1 && $7 != "" && $7 != "None" {s+=$7; n++} END {if(n>0) printf "%.4f", s/n; else print "N/A"}' "$d/accuracy.csv")
        echo "n_total=$n_total  n_scored=$n_scored  avg_score=$avg"
    fi
    echo ""
done

cat <<EOF

════════════════════════════════════════════════════════════════
快速对比 ttft_p50 跨 3 组:
  for d in $EXP_ROOT/*/; do
    name=\$(basename \$d)
    echo "=== \$name ==="
    awk -F, 'NR>1 {printf "turn=%s  prompt=%s  ttft_p50=%s  e2e_p50=%s\n", \$1, \$3, \$4, \$13}' \$d/turn_summary.csv
  done

吞吐对比 (各组总耗时):
  for d in $EXP_ROOT/*/; do
    name=\$(basename \$d)
    total=\$(awk '{sum += \$1} END {print sum}' < <(jq '.total_e2e_ms' \$d/samples.jsonl))
    echo "\$name: total_e2e=\${total}ms (越小说明吞吐越高)"
  done
════════════════════════════════════════════════════════════════
EOF
