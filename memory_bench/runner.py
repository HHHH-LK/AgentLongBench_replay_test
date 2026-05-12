"""编排层 + CLI: 按样本逐轮回放, 写出 turns.jsonl + samples.jsonl。

KV 复用纪律 (绝对不要破坏):
  - 同一 sample 的 N 轮请求必须**串行**打到**同一个 base_url**
  - 同一 sample 内不能并行
  - 多 sample 之间可以 --workers N 并行 (它们 prefix 不同, 不影响 KV 命中)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .io_utils import (
    append_jsonl,
    infer_context_from_path,
    load_jsonl,
    require_single_question_type,
)
from .parsing import parse_response
from .prompts import build_system_prompt
from .replay import split_into_turns
from .scoring import score_one
from .timer import timed_chat
from .types import Backend, QuestionType, SampleResult, TurnMetrics, TurnStatus


# ════════════════════════════════════════════════════════════════════════
# 单 sample 处理
# ════════════════════════════════════════════════════════════════════════

def _process_sample(
    sample: Dict[str, Any],
    *,
    qt: QuestionType,
    knowledge_label: str,
    history_label: str,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    max_turns: Optional[int],
    extra_payload: Optional[Dict[str, Any]] = None,
) -> Tuple[List[TurnMetrics], SampleResult]:
    """跑一条样本的所有轮次 (串行, 同一 base_url)。"""
    sys_prompt = build_system_prompt(qt, knowledge_label, history_label)
    turns = split_into_turns(sample, sys_prompt, max_turns=max_turns)

    sid = str(sample.get("id"))
    turn_records: List[TurnMetrics] = []
    cum_tokens = 0
    total_e2e = 0.0
    final_raw: Optional[str] = None
    final_pt: Optional[int] = None

    for t in turns:
        result = timed_chat(
            base_url=base_url,
            model=model,
            messages=t.request_messages,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            extra_payload=extra_payload,
        )
        if result.prompt_tokens is not None:
            cum_tokens = max(cum_tokens, result.prompt_tokens)
        total_e2e += result.e2e_ms or 0.0
        if t.is_final:
            final_raw = result.raw_response
            final_pt = result.prompt_tokens

        if result.error:
            print(f"  [turn-err] {t.id}: {result.error}", flush=True)
        turn_records.append(TurnMetrics(
            turn_id=t.id,
            sample_id=sid,
            turn_index=t.index,
            backend=Backend.API.value,
            model_name=model,
            is_warmup=t.is_warmup,
            is_final=t.is_final,
            ttft_ms=result.ttft_ms,
            e2e_ms=result.e2e_ms,
            decode_ms=result.decode_ms,
            tpot_ms=result.tpot_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cumulative_prompt_tokens=cum_tokens or None,
            raw_response=result.raw_response if t.is_final else None,
            status=(TurnStatus.ERROR.value if result.error else TurnStatus.OK.value),
            error=result.error,
        ))

    pred, parse_kind = (None, None)
    if final_raw:
        pred, parse_kind = parse_response(qt, history_label, final_raw)
    correct = score_one(qt, history_label, knowledge_label, pred, sample.get("answer"))

    sample_result = SampleResult(
        sample_id=sid,
        backend=Backend.API.value,
        model_name=model,
        question_type=qt.value,
        knowledge_label=knowledge_label,
        history_label=history_label,
        num_turns=len(turn_records),
        total_e2e_ms=total_e2e,
        final_prompt_tokens=final_pt,
        raw_response=final_raw,
        pred_answer=pred,
        parse_kind=parse_kind,
        gold_answer=sample.get("answer"),
        correct=correct,
        extra={k: sample[k] for k in ("round", "i_round", "j_round") if k in sample},
    )
    return turn_records, sample_result


# ════════════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════════════

def run(
    dataset_path: Path,
    output_dir: Path,
    *,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    timeout: int = 1200,
    offset: int = 0,
    limit: Optional[int] = None,
    max_turns: Optional[int] = None,
    workers: int = 1,
    run_id: Optional[str] = None,
    disable_thinking: bool = False,
) -> Dict[str, Any]:
    rows = load_jsonl(dataset_path)
    rows = rows[offset:] if limit is None else rows[offset : offset + limit]

    qt_raw = require_single_question_type(rows)
    qt = QuestionType.from_any(qt_raw)
    knowledge_label, history_label = infer_context_from_path(dataset_path)

    run_id = run_id or f"{int(time.time())}-{uuid.uuid4().hex[:6]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    turns_path = output_dir / "turns.jsonl"
    samples_path = output_dir / "samples.jsonl"
    # 重置文件 (避免 append 到旧数据上)
    turns_path.write_text("", encoding="utf-8")
    samples_path.write_text("", encoding="utf-8")

    # 关闭 Qwen Thinking 模式 (vLLM chat_template 透传)
    extra_payload: Optional[Dict[str, Any]] = None
    if disable_thinking:
        extra_payload = {"chat_template_kwargs": {"enable_thinking": False}}

    print(
        f"[run] run_id={run_id}\n"
        f"      dataset={dataset_path.name}  samples={len(rows)}\n"
        f"      qtype={qt.value}  knowledge={knowledge_label}  history={history_label}\n"
        f"      backend=api  model={model}  base_url={base_url}\n"
        f"      workers={workers}  disable_thinking={disable_thinking}",
        flush=True,
    )

    def _do_one(sample: Dict[str, Any]):
        return _process_sample(
            sample,
            qt=qt,
            knowledge_label=knowledge_label,
            history_label=history_label,
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_turns=max_turns,
            extra_payload=extra_payload,
        )

    n_done = 0
    n_correct = 0.0
    n_scored = 0

    if workers <= 1:
        for i, sample in enumerate(rows, 1):
            try:
                turn_recs, sample_res = _do_one(sample)
            except Exception as e:
                print(f"[err] sample {sample.get('id')} failed: {e}", file=sys.stderr)
                continue
            for tr in turn_recs:
                append_jsonl(turns_path, tr.to_dict())
            append_jsonl(samples_path, sample_res.to_dict())
            n_done += 1
            if sample_res.correct is not None:
                n_correct += sample_res.correct
                n_scored += 1
            avg = (n_correct / n_scored) if n_scored else 0.0
            print(
                f"[{i}/{len(rows)}] sid={sample_res.sample_id} "
                f"turns={sample_res.num_turns} "
                f"e2e={sample_res.total_e2e_ms:.0f}ms "
                f"correct={sample_res.correct} "
                f"running_acc={avg:.3f}",
                flush=True,
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_do_one, s): s for s in rows}
            for fut in as_completed(futures):
                sample = futures[fut]
                try:
                    turn_recs, sample_res = fut.result()
                except Exception as e:
                    print(f"[err] sample {sample.get('id')} failed: {e}", file=sys.stderr)
                    continue
                for tr in turn_recs:
                    append_jsonl(turns_path, tr.to_dict())
                append_jsonl(samples_path, sample_res.to_dict())
                n_done += 1
                if sample_res.correct is not None:
                    n_correct += sample_res.correct
                    n_scored += 1
                avg = (n_correct / n_scored) if n_scored else 0.0
                print(
                    f"[{n_done}/{len(rows)}] sid={sample_res.sample_id} "
                    f"turns={sample_res.num_turns} "
                    f"e2e={sample_res.total_e2e_ms:.0f}ms "
                    f"correct={sample_res.correct} "
                    f"running_acc={avg:.3f}",
                    flush=True,
                )

    summary = {
        "run_id": run_id,
        "dataset": str(dataset_path),
        "n_done": n_done,
        "n_scored": n_scored,
        "accuracy": (n_correct / n_scored) if n_scored else None,
        "turns_path": str(turns_path),
        "samples_path": str(samples_path),
    }
    print(f"\n[done] {summary}", flush=True)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Vanilla 多轮 KV-reuse 性能基线 (OpenAI 兼容 API)")
    p.add_argument("--dataset", type=Path, required=True, help="AgentLongBench .jsonl")
    p.add_argument("--output-dir", type=Path, required=True, help="目录, 写 turns/samples.jsonl")
    p.add_argument("--base-url", type=str, required=True, help="如 http://server:17100/v1")
    p.add_argument("--model", type=str, required=True, help="--served-model-name 那个名字")
    p.add_argument("--api-key", type=str, default=os.environ.get("API_KEY", "EMPTY"))
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--timeout", type=int, default=1200)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-turns", type=int, default=None)
    p.add_argument("--workers", type=int, default=1, help="跨样本并行数 (同样本内永远串行)")
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--disable-thinking", action="store_true",
                   help="关闭 Qwen3-Thinking 的思考段 (传 chat_template_kwargs.enable_thinking=false)")
    args = p.parse_args()

    run(
        args.dataset,
        args.output_dir,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        offset=args.offset,
        limit=args.limit,
        max_turns=args.max_turns,
        workers=args.workers,
        run_id=args.run_id,
        disable_thinking=args.disable_thinking,
    )


if __name__ == "__main__":
    main()
