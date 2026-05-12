#!/usr/bin/env bash
# 三组对照实验:
#   01  LMCache ON   workers=1            (KV 复用纯净基线)
#   02  LMCache OFF  workers=1            (对照组, 无 KV 复用)
#   03  LMCache ON   workers=$WORKERS_MULTI  (多并发 + KV 复用, 看吞吐)
#
# 每组之间需要在远程机器重启 vLLM (脚本会显示完整启动命令, 你粘贴执行后回车继续)
#
# 用法:
#   export DATASET=/path/to/intersection.jsonl
#   export LIMIT=5                   # 跑前 N 条样本
#   export WORKERS_MULTI=4           # 第 3 组的 workers 数
#   export BASE_URL=http://10.26.6.54:17100/v1
#   export MODEL=qwen3-30b-thinking
#   ./scripts/run_3_experiments.sh

set -euo pipefail

# ────────────────────────────────────────────────────────────────
# 配置 (可被环境变量覆盖)
# ────────────────────────────────────────────────────────────────
: "${DATASET:?需要 export DATASET=/path/to/dataset.jsonl}"
: "${BASE_URL:=http://10.26.6.54:17100/v1}"
: "${MODEL:=qwen3-30b-thinking}"
: "${API_KEY:=EMPTY}"
: "${LIMIT:=5}"
: "${WORKERS_MULTI:=4}"
: "${MAX_TOKENS:=2048}"
: "${TEMPERATURE:=0.6}"
: "${DISABLE_THINKING:=1}"            # 1=关 thinking, 0=保留
: "${WAIT_READY_SEC:=1800}"           # vLLM 就绪等待 (默认 30 分钟)
: "${LMCACHE_SIZE_GB:=50}"            # LMCache CPU 池大小

ROOT=$(cd "$(dirname "$0")/.." && pwd)
TS=$(date +%Y%m%d_%H%M%S)
EXP_ROOT="$ROOT/runs/exp_${TS}"
mkdir -p "$EXP_ROOT"

# ────────────────────────────────────────────────────────────────
# 工具函数
# ────────────────────────────────────────────────────────────────
log()   { echo -e "\n\033[1;36m[$(date +%H:%M:%S)] $*\033[0m"; }
info()  { echo -e "\033[0;33m  → $*\033[0m"; }
ok()    { echo -e "\033[0;32m  ✓ $*\033[0m"; }
err()   { echo -e "\033[0;31m  ✗ $*\033[0m" >&2; }

wait_until_ready() {
    local timeout_sec=$1
    local end=$(( $(date +%s) + timeout_sec ))
    info "等待 vLLM 就绪 (探测 $BASE_URL/models, 最长 ${timeout_sec}s)..."
    local last_code=""
    while [[ $(date +%s) -lt $end ]]; do
        local code
        code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 \
               "$BASE_URL/models" 2>/dev/null || echo 000)
        if [[ "$code" == "200" ]]; then
            echo ""
            ok "vLLM 已就绪 (http_code=200)"
            return 0
        fi
        if [[ "$code" != "$last_code" ]]; then
            echo -n " [code=$code]"
            last_code=$code
        else
            echo -n "."
        fi
        sleep 10
    done
    echo ""
    err "等待超时 (${timeout_sec}s), vLLM 没起来"
    return 1
}

print_vllm_command() {
    local mode=$1   # lmcache_on / lmcache_off
    local logfile="vllm_${mode}.log"
    cat <<EOF

────────────────────────────────────────────────────────────────
请在远程机器执行 (复制下面整段)
────────────────────────────────────────────────────────────────

# Step 1: 清理旧进程
docker exec lk_hhhh bash -c '
  for p in \$(ps -ef | grep -E "vllm|api_server" | grep -v grep | awk "{print \\\$2}"); do
    kill -9 \$p 2>/dev/null
  done
  sleep 10
  echo "[clean] NPU 状态:"
  npu-smi info | grep -E "910B3.*HBM|No running" | head -10
'

# Step 2: 启动新 vLLM ($mode)
EOF

    if [[ $mode == lmcache_on ]]; then
        cat <<EOF
docker exec -d lk_hhhh bash -lc '
cd /home
export ASCEND_RT_VISIBLE_DEVICES=3,4,5,6
export PYTHONHASHSEED=0
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_CONNECT_TIMEOUT=1800
export MASTER_PORT=29631
export LMCACHE_CHUNK_SIZE=256
export LMCACHE_LOCAL_CPU=true
export LMCACHE_MAX_LOCAL_CPU_SIZE=$LMCACHE_SIZE_GB
export LMCACHE_CACHE_POLICY=LRU
export LMCACHE_USE_LAYERWISE=false

python -m vllm.entrypoints.openai.api_server \\
  --host 0.0.0.0 --port 17100 \\
  --model /home/models/Qwen3-30B-A3B-Thinking-2507 \\
  --served-model-name qwen3-30b-thinking \\
  --trust-remote-code --disable-log-requests \\
  --block-size 128 \\
  --tensor-parallel-size 4 \\
  --max-model-len 131072 \\
  --max-num-batched-tokens 2048 \\
  --gpu-memory-utilization 0.9 \\
  --enforce-eager \\
  --kv-transfer-config '\''{"kv_connector":"LMCacheAscendConnectorV1Dynamic","kv_role":"kv_both","kv_connector_module_path":"lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"}'\'' \\
  > /home/$logfile 2>&1
'
EOF
    else
        cat <<EOF
docker exec -d lk_hhhh bash -lc '
cd /home
export ASCEND_RT_VISIBLE_DEVICES=3,4,5,6
export PYTHONHASHSEED=0
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_CONNECT_TIMEOUT=1800
export MASTER_PORT=29631

python -m vllm.entrypoints.openai.api_server \\
  --host 0.0.0.0 --port 17100 \\
  --model /home/models/Qwen3-30B-A3B-Thinking-2507 \\
  --served-model-name qwen3-30b-thinking \\
  --trust-remote-code --disable-log-requests \\
  --block-size 128 \\
  --tensor-parallel-size 4 \\
  --max-model-len 131072 \\
  --max-num-batched-tokens 2048 \\
  --gpu-memory-utilization 0.9 \\
  --enforce-eager \\
  > /home/$logfile 2>&1
'
EOF
    fi

    cat <<EOF

────────────────────────────────────────────────────────────────
执行完命令后按回车继续 (脚本会自动 curl 探活并等待就绪)
EOF
}

prompt_and_wait() {
    local mode=$1
    log "═══ 准备阶段: vLLM 切换到 [$mode] ═══"
    print_vllm_command "$mode"
    read -r
    wait_until_ready "$WAIT_READY_SEC" || exit 1
}

run_phase() {
    local name=$1
    local workers=$2
    local out="$EXP_ROOT/$name"

    log "═══ 跑实验 [$name] workers=$workers ═══"
    info "输出目录: $out"

    local extra=()
    [[ "$DISABLE_THINKING" == "1" ]] && extra+=(--disable-thinking)

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
        "${extra[@]}" 2>&1 | tee "$out.runner.log"; then
        ok "[$name] runner 完成"
    else
        err "[$name] runner 失败, 见 $out.runner.log"
        return 1
    fi

    uv run python -m memory_bench.analyze --run-dir "$out" 2>&1 | tee "$out.analyze.log"
    ok "[$name] 完成 → $out"
}

# ────────────────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────────────────
log "≡≡≡ 三组对照实验 ≡≡≡"
info "数据集: $DATASET"
info "样本数: $LIMIT"
info "多并发 workers: $WORKERS_MULTI"
info "max_tokens: $MAX_TOKENS  temperature: $TEMPERATURE  disable_thinking: $DISABLE_THINKING"
info "服务地址: $BASE_URL"
info "结果目录: $EXP_ROOT"
echo ""

# ─── 实验 1: LMCache ON, workers=1 ───
prompt_and_wait lmcache_on
run_phase "01_lmcache_on_w1" 1

# ─── 实验 2: LMCache OFF, workers=1 ───
prompt_and_wait lmcache_off
run_phase "02_lmcache_off_w1" 1

# ─── 实验 3: LMCache ON, workers=N ───
prompt_and_wait lmcache_on
run_phase "03_lmcache_on_w${WORKERS_MULTI}" "$WORKERS_MULTI"

# ────────────────────────────────────────────────────────────────
# 汇总
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
        # 只看最后一行汇总
        tail -1 "$d/accuracy.csv"
    fi
    echo ""
done

cat <<EOF

════════════════════════════════════════════════════════════════
快速对比命令:

  diff $EXP_ROOT/01_lmcache_on_w1/turn_summary.csv \\
       $EXP_ROOT/02_lmcache_off_w1/turn_summary.csv

  对比 ttft_ms_p50 (第 4 列):
  for d in $EXP_ROOT/*/; do
    name=\$(basename \$d)
    echo "=== \$name ==="
    awk -F, 'NR>1 {printf "turn=%s  prompt=%s  ttft=%s\n", \$1, \$3, \$4}' \$d/turn_summary.csv
  done
════════════════════════════════════════════════════════════════
EOF
