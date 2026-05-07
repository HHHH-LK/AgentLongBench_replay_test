#!/usr/bin/env bash
# 仅做分析 (已经有 turns.jsonl + samples.jsonl 时用)
#
# 用法: ./scripts/analyze.sh <run_dir>

set -euo pipefail

RUN_DIR=${1:?"用法: $0 <run_dir>"}
python -m memory_bench.analyze --run-dir "$RUN_DIR"
