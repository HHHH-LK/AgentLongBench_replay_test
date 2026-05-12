# LongMemEval Replay

This folder contains a standalone replay-mode runner for LongMemEval.

It is different from the official LongMemEval one-shot evaluation:

- official style: selected history chats + question are sent in one request;
- replay style: selected sessions are appended one by one, every prefix is sent with streaming enabled, and the final request appends the question.

Intermediate model outputs are discarded. They are only used to measure TTFT/E2E/TPOT at each growing context length.

## Quick Start

```bash
uv run python -m longmemeval_replay.runner \
  --dataset /path/to/longmemeval_s_cleaned.json \
  --output-dir runs/longmemeval_replay_smoke \
  --base-url http://10.26.6.54:17100/v1 \
  --model qwen3-30b-instruct \
  --topk-context 3 \
  --limit 1 \
  --max-tokens 512 \
  --temperature 0 \
  --disable-thinking
```

Outputs:

```text
turns.jsonl    # one row per replay turn, includes ttft_ms/e2e_ms/prompt_tokens
hypotheses.jsonl # official-compatible predictions: {"question_id", "hypothesis"}
samples.jsonl  # one row per sample, includes final answer and optional judge label
eval_results.jsonl # only when --judge-model is set
summary.json   # aggregate TTFT and accuracy summary
```

## Notes

- `--topk-context` follows the official LongMemEval `orig-session` behavior: it uses the latest N sessions.
- `--history-format json` mirrors the official script's JSON history format.
- Accuracy follows the official LongMemEval evaluator style when `--judge-model` is set: it builds the same question-type-specific yes/no judge prompt and stores the judge label.
- Without `--judge-model`, this runner does not guess correctness. It writes `hypotheses.jsonl`, which can be passed to the official `src/evaluation/evaluate_qa.py`.

## Inline Official-Style Judge

```bash
uv run python -m longmemeval_replay.runner \
  --dataset /path/to/longmemeval_s_cleaned.json \
  --output-dir runs/longmemeval_replay_judged \
  --base-url http://10.26.6.54:17100/v1 \
  --model qwen3-30b-instruct \
  --judge-base-url http://10.26.6.54:17100/v1 \
  --judge-model qwen3-30b-instruct \
  --topk-context 3 \
  --limit 1 \
  --max-tokens 512 \
  --temperature 0 \
  --disable-thinking
```

For strict paper-style reporting, use a stronger judge model and/or run the official evaluator on `hypotheses.jsonl`.
