# MemoyAgentTest

**Vanilla 多轮 Agent 会话 KV 复用性能基线** — 用 AgentLongBench 数据集模拟一个真实 agent 场景，按轮累积 prefix 打到 vLLM serve，采集每一轮的 TTFT / TPOT / E2E / token 数等性能指标，同时统计每条样本的回答准确率。

后续做任何长上下文优化（KV 卸载、上下文压缩、prefill 切分等）的对照组就用这份基线。

## 设计核心

- **Agent 场景模拟**：把数据集里录好的多轮对话，当作"如果当时一轮一轮真打过来"回放
- **每轮请求 = 历史 prefix + 当前新消息**：累积式而非一次性灌入，让推理引擎的 prefix KV cache 自然命中
- **同一 sample 严格串行打到同一 backend**：保证 KV 命中条件
- **前 N-1 轮模型输出丢弃**：只采指标；下一轮 prefix 用数据集里录好的 assistant/tool 接上（保证可复现）
- **最后一轮保留输出**：跟 gold answer 对比统计准确率

## 项目结构

```
MemoyAgentTest/
├── memory_bench/
│   ├── types.py        枚举 + dataclass (Turn / TurnMetrics / SampleResult)
│   ├── prompts.py      AgentLongBench 8 题型的标准 system prompt
│   ├── parsing.py      模型输出 → pred_answer (number/boolean/list/...)
│   ├── scoring.py      pred vs gold → 0/1 或 F1
│   ├── io_utils.py     JSONL 读写 + ki-c/kf-v 路径推断
│   ├── replay.py       切分层 (sample → N 轮累积 prefix)
│   ├── timer.py        计时层 (单次 SSE 流式请求)
│   ├── runner.py       编排层 + CLI
│   └── analyze.py      P50/P90/P99 + 准确率聚合
├── scripts/
│   ├── run_baseline.sh
│   └── analyze.sh
└── runs/               # 输出目录, 每次 run 一个子目录
```

## 安装

```bash
cd MemoyAgentTest
uv sync                         # 或 pip install -e .
```

只需要一个外部依赖：`requests`。

## 启动 vLLM serve（前置）

确保你的 vLLM 服务已就绪 (本项目走 OpenAI 兼容 HTTP)：

```bash
curl -s http://<server>:17100/v1/models
# 期望返回 JSON 含模型名
```

## 烟测（强烈建议先跑这一步）

```bash
# 1. replay 自检 (无需服务)
python -m memory_bench.replay
# 期望: [OK] selftest passed, 4 turns generated

# 2. 单样本端到端 (需要 vLLM serve 已启动)
export BASE_URL=http://<server>:17100/v1
export MODEL=qwen3-30b-thinking
./scripts/run_baseline.sh path/to/intersection.jsonl runs/smoke 1

# 3. 检查产物
head -1 runs/smoke/turns.jsonl   | python -m json.tool
head -1 runs/smoke/samples.jsonl | python -m json.tool
cat runs/smoke/turn_summary.csv
```

期望看到：
- `turns.jsonl` 行数 = 该样本轮数
- `ttft_ms` / `e2e_ms` 都是合理毫秒数
- `cumulative_prompt_tokens` 随 `turn_index` 单调上升

## 正式跑基线

```bash
./scripts/run_baseline.sh \
    /path/to/AgentLongBench/benchmark/ki-c/128k/final_guess/intersection.jsonl \
    runs/baseline_lmcache_on_$(date +%m%d)
```

或直接调 runner：

```bash
python -m memory_bench.runner \
    --dataset path/to/intersection.jsonl \
    --output-dir runs/baseline_001 \
    --base-url http://<server>:17100/v1 \
    --model qwen3-30b-thinking \
    --workers 1
```

## 产物

每次跑会得到一个 run 目录：

```
runs/baseline_xxxx/
├── turns.jsonl           每行一条 TurnMetrics (一个 sample 的一轮)
├── samples.jsonl         每行一条 SampleResult (一个 sample 的最终结果)
├── turn_summary.csv      按 turn_index 聚合 P50/P90/P99
├── bucket_summary.csv    按 prompt_tokens 分桶聚合
└── accuracy.csv          每条样本对错明细 + 总分
```

## CLI 参数

```
python -m memory_bench.runner \
    --dataset PATH        AgentLongBench .jsonl
    --output-dir PATH     run 目录
    --base-url URL        如 http://server:17100/v1
    --model NAME          --served-model-name 那个名字
    --api-key KEY         本地服务可写 EMPTY
    --temperature F       默认 0.7
    --max-tokens N        默认 1024
    --timeout S           默认 1200
    --offset N            从第 N 条开始
    --limit N             只跑前 N 条
    --max-turns K         每条样本最多 K 轮
    --workers N           跨样本并行数 (同样本内永远串行)
```

## KV 复用纪律（不要破坏！）

```
✅ 同一 sample 的 N 轮请求必须串行打到同一个 base_url
✅ 多 sample 之间可以 --workers 并行
❌ 不要在 sample 内并行
❌ 不要让模型实时输出影响下一轮 prefix (必须用数据集录好的)
❌ 跑两组对照实验 (优化 ON/OFF) 中间必须重启 vLLM 服务清掉 KV
```

## 后续做优化对比怎么对接

后面你想测某个优化（压缩、KV 卸载等）相对 baseline 的收益：

1. 用同一份 dataset、同一个 base_url 配置
2. 跑两遍 runner，分别跑在"优化关闭" vs "优化开启"两个 vLLM 实例上
3. 用 `analyze` 各自出 `turn_summary.csv`
4. 同 `turn_index` 行对应位置直接 diff TTFT P50

`TurnMetrics` 已经预留了 `compression_time_ms` / `compression_ratio` 字段，未来加压缩钩子时直接填即可，schema 不用改。

## 数据集格式（AgentLongBench）

每行一条 sample：

```jsonc
{
  "id": "...",
  "sample_id": "...",
  "question_type": "Intersection",          // 或 slug 形式
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "content": "..."},
    ...
  ],
  "question": "最终要问的问题",
  "answer": "标准答案"                        // 数值/布尔/列表/字符串
}
```

数据集路径里需要含一段 `ki-c` / `ki-v` / `kf-c` / `kf-v` 来标识知识/历史风格，例如 `.../ki-c/128k/final_guess/intersection.jsonl`。

## 支持的题型

8 种 AgentLongBench 题型全部支持准确率打分（详见 `scoring.py`）：
- Tool Response: `count_frequency_tool` / `find_duplicates_tool` / `find_target_offsets_tool`
- Env Response: `count_correctness_env` / `count_frequency_env` / `find_round_largest_value_env` / `weighted_summation_env`
- Final Guess: `intersection`（含 Verbose 版的 set F1）
