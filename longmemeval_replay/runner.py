"""LongMemEval replay-mode benchmark.

This runner adapts LongMemEval into a session replay workload:
  - each historical session is appended as one new user message;
  - every prefix is sent to an OpenAI-compatible chat API with streaming on;
  - intermediate responses are discarded except for timing;
  - the final turn appends the LongMemEval question and records the answer.

The prompt wording follows the official LongMemEval generation prompt closely:
"I will give you several history chats..." + "History Chats" + "Current Date"
+ "Question" + "Answer".
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests


SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question based only on the "
    "provided history chats. If the answer cannot be found in the history, say "
    "that the information is not available."
)

HISTORY_HEADER = (
    "I will give you several history chats between you and a user.\n"
    "Please answer the question based on the relevant chat history.\n\n"
    "History Chats:"
)

FINAL_QUESTION_TEMPLATE = (
    "Current Date: {question_date}\n"
    "Question: {question}\n"
    "Answer:"
)


@dataclass
class ReplayTurn:
    sample_id: str
    index: int
    messages: List[Dict[str, str]]
    is_final: bool
    is_warmup: bool


@dataclass
class TimedResult:
    raw_response: str
    ttft_ms: Optional[float]
    e2e_ms: Optional[float]
    decode_ms: Optional[float]
    tpot_ms: Optional[float]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    error: Optional[str] = None


def load_dataset(path: str | Path, *, offset: int = 0, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load LongMemEval JSON-array files or JSONL files."""
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        rows: List[Dict[str, Any]] = []
    elif text[0] == "[":
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return rows[offset:] if limit is None else rows[offset : offset + limit]


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def strip_internal_fields(turn: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in turn.items() if k != "has_answer"}


def format_session(session_index: int, session_date: str, session: Any, *, history_format: str) -> str:
    if isinstance(session, list):
        cleaned = [
            strip_internal_fields(msg) if isinstance(msg, dict) else msg
            for msg in session
        ]
    else:
        cleaned = strip_internal_fields(session) if isinstance(session, dict) else session

    if history_format == "json":
        body = json.dumps(cleaned, ensure_ascii=False)
    elif history_format == "nl":
        if isinstance(cleaned, list):
            parts = []
            for msg in cleaned:
                if isinstance(msg, dict):
                    role = msg.get("role", "unknown")
                    content = str(msg.get("content", "")).strip()
                    parts.append(f"{role}: {content}")
                else:
                    parts.append(str(msg))
            body = "\n".join(parts)
        elif isinstance(cleaned, dict):
            body = f"{cleaned.get('role', 'unknown')}: {str(cleaned.get('content', '')).strip()}"
        else:
            body = str(cleaned)
    else:
        raise ValueError(f"unsupported history_format: {history_format}")

    return (
        f"### Session {session_index}:\n"
        f"Session Date: {session_date}\n"
        f"Session Content:\n{body}"
    )


def select_sessions(entry: Dict[str, Any], *, topk_context: int, user_only: bool) -> List[tuple[str, Any]]:
    dates = entry.get("haystack_dates") or []
    sessions = entry.get("haystack_sessions") or []
    selected = list(zip(dates, sessions))[-topk_context:]
    if not user_only:
        return selected
    user_selected = []
    for date, session in selected:
        if isinstance(session, list):
            user_selected.append((date, [m for m in session if isinstance(m, dict) and m.get("role") == "user"]))
        else:
            user_selected.append((date, session))
    return user_selected


def build_replay_turns(
    entry: Dict[str, Any],
    *,
    topk_context: int,
    history_format: str = "json",
    user_only: bool = False,
) -> List[ReplayTurn]:
    """Build one request per appended LongMemEval session plus one final QA turn."""
    sample_id = str(entry.get("question_id") or entry.get("id") or entry.get("sample_id"))
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": HISTORY_HEADER},
    ]
    turns: List[ReplayTurn] = []

    sessions = select_sessions(entry, topk_context=topk_context, user_only=user_only)
    for idx, (session_date, session) in enumerate(sessions, 1):
        messages.append({
            "role": "user",
            "content": format_session(idx, str(session_date), session, history_format=history_format),
        })
        turns.append(ReplayTurn(
            sample_id=sample_id,
            index=len(turns),
            messages=[dict(m) for m in messages],
            is_final=False,
            is_warmup=(len(turns) == 0),
        ))

    question_date = entry.get("question_date") or entry.get("date") or "Unknown"
    messages.append({
        "role": "user",
        "content": FINAL_QUESTION_TEMPLATE.format(
            question_date=question_date,
            question=entry.get("question", ""),
        ),
    })
    turns.append(ReplayTurn(
        sample_id=sample_id,
        index=len(turns),
        messages=[dict(m) for m in messages],
        is_final=True,
        is_warmup=(len(turns) == 0),
    ))
    return turns


def timed_chat(
    *,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    api_key: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    disable_thinking: bool,
) -> TimedResult:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    t0 = time.perf_counter()
    first_token_at: Optional[float] = None
    content: List[str] = []
    usage: Dict[str, Any] = {}
    error: Optional[str] = None

    try:
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8")
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content") or ""
                if piece:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    content.append(piece)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    e2e_ms = (time.perf_counter() - t0) * 1000.0
    ttft_ms = (first_token_at - t0) * 1000.0 if first_token_at else None
    decode_ms = max(e2e_ms - ttft_ms, 0.0) if ttft_ms is not None else None
    completion_tokens = usage.get("completion_tokens")
    tpot_ms = None
    if decode_ms is not None and completion_tokens and completion_tokens > 1:
        tpot_ms = decode_ms / max(completion_tokens - 1, 1)

    return TimedResult(
        raw_response="".join(content),
        ttft_ms=ttft_ms,
        e2e_ms=e2e_ms,
        decode_ms=decode_ms,
        tpot_ms=tpot_ms,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=completion_tokens,
        error=error,
    )


def get_anscheck_prompt(
    task: str,
    question: str,
    answer: str,
    response: str,
    *,
    abstention: bool = False,
) -> str:
    """Build the same yes/no judge prompt shape as LongMemEval's official evaluator."""
    if abstention:
        return (
            "I will give you an unanswerable question, an explanation, and a response "
            "from a model. Please answer yes if the model correctly identifies the "
            "question as unanswerable. The model could say that the information is "
            "incomplete, or some other information is given but the asked information "
            "is not.\n\n"
            f"Question: {question}\n\n"
            f"Explanation: {answer}\n\n"
            f"Model Response: {response}\n\n"
            "Does the model correctly identify the question as unanswerable? "
            "Answer yes or no only."
        )

    if task in {"single-session-user", "single-session-assistant", "multi-session"}:
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, "
            "answer no. If the response is equivalent to the correct answer or contains "
            "all the intermediate steps to get the correct answer, you should also "
            "answer yes. If the response only contains a subset of the information "
            "required by the answer, answer no.\n\n"
            f"Question: {question}\n\n"
            f"Correct Answer: {answer}\n\n"
            f"Model Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    if task == "temporal-reasoning":
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, "
            "answer no. If the response is equivalent to the correct answer or contains "
            "all the intermediate steps to get the correct answer, you should also "
            "answer yes. If the response only contains a subset of the information "
            "required by the answer, answer no. In addition, do not penalize off-by-one "
            "errors for the number of days. If the question asks for the number of "
            "days/weeks/months, etc., and the model makes off-by-one errors, the "
            "model's response is still correct.\n\n"
            f"Question: {question}\n\n"
            f"Correct Answer: {answer}\n\n"
            f"Model Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    if task == "knowledge-update":
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, "
            "answer no. If the response contains some previous information along with "
            "an updated answer, the response should be considered as correct as long "
            "as the updated answer is the required answer.\n\n"
            f"Question: {question}\n\n"
            f"Correct Answer: {answer}\n\n"
            f"Model Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    if task == "single-session-preference":
        return (
            "I will give you a question, a rubric for desired personalized response, "
            "and a response from a model. Please answer yes if the response satisfies "
            "the desired response. Otherwise, answer no. The model does not need to "
            "reflect all the points in the rubric. The response is correct as long as "
            "it recalls and utilizes the user's personal information correctly.\n\n"
            f"Question: {question}\n\n"
            f"Rubric: {answer}\n\n"
            f"Model Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    raise NotImplementedError(f"unsupported LongMemEval question_type: {task!r}")


def judge_response(
    *,
    base_url: Optional[str],
    model: Optional[str],
    api_key: str,
    question_type: str,
    question_id: str,
    question: str,
    answer: str,
    hypothesis: str,
    timeout: int,
) -> tuple[Optional[bool], Optional[str], Optional[str]]:
    """Run the official LongMemEval yes/no judge prompt when a judge model is configured."""
    if not model:
        return None, None, None
    if not base_url:
        return None, None, "judge_base_url is required when judge_model is set"
    prompt = get_anscheck_prompt(
        question_type,
        question,
        answer,
        hypothesis,
        abstention="_abs" in question_id,
    )
    result = timed_chat(
        base_url=base_url,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        api_key=api_key,
        temperature=0,
        max_tokens=10,
        timeout=timeout,
        disable_thinking=False,
    )
    if result.error:
        return None, result.raw_response, result.error
    label = "yes" in result.raw_response.lower()
    return label, result.raw_response, None


def percentile(values: Iterable[Optional[float]], q: float) -> Optional[float]:
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    k = (len(clean) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(clean) - 1)
    return clean[lo] if lo == hi else clean[lo] + (clean[hi] - clean[lo]) * (k - lo)


def summarize(turn_rows: List[Dict[str, Any]], sample_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok_turns = [r for r in turn_rows if r.get("status") == "ok"]
    final_turns = [r for r in ok_turns if r.get("is_final")]
    scored = [r for r in sample_rows if r.get("correct") is not None]
    return {
        "n_samples": len(sample_rows),
        "n_turns": len(turn_rows),
        "n_ok_turns": len(ok_turns),
        "accuracy": (
            sum(r["correct"] for r in scored) / len(scored)
            if scored else None
        ),
        "final_ttft_p50_ms": percentile((r.get("ttft_ms") for r in final_turns), 0.5),
        "final_ttft_p90_ms": percentile((r.get("ttft_ms") for r in final_turns), 0.9),
        "turn_ttft_p50_ms": percentile((r.get("ttft_ms") for r in ok_turns), 0.5),
        "turn_ttft_p90_ms": percentile((r.get("ttft_ms") for r in ok_turns), 0.9),
        "avg_final_prompt_tokens": (
            statistics.mean(r["prompt_tokens"] for r in final_turns if r.get("prompt_tokens"))
            if any(r.get("prompt_tokens") for r in final_turns) else None
        ),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    rows = load_dataset(args.dataset, offset=args.offset, limit=args.limit)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    turns_path = output_dir / "turns.jsonl"
    samples_path = output_dir / "samples.jsonl"
    hypotheses_path = output_dir / "hypotheses.jsonl"
    eval_results_path = output_dir / "eval_results.jsonl"
    summary_path = output_dir / "summary.json"
    turns_path.write_text("", encoding="utf-8")
    samples_path.write_text("", encoding="utf-8")
    hypotheses_path.write_text("", encoding="utf-8")
    eval_results_path.write_text("", encoding="utf-8")

    print(
        f"[run] dataset={args.dataset} samples={len(rows)} mode=longmemeval-replay\n"
        f"      model={args.model} base_url={args.base_url}\n"
        f"      topk_context={args.topk_context} history_format={args.history_format} "
        f"user_only={args.user_only} disable_thinking={args.disable_thinking}",
        flush=True,
    )

    all_turn_rows: List[Dict[str, Any]] = []
    all_sample_rows: List[Dict[str, Any]] = []
    for i, entry in enumerate(rows, 1):
        sample_id = str(entry.get("question_id") or entry.get("id") or i)
        turns = build_replay_turns(
            entry,
            topk_context=args.topk_context,
            history_format=args.history_format,
            user_only=args.user_only,
        )
        total_e2e_ms = 0.0
        final_raw = ""
        final_prompt_tokens: Optional[int] = None
        sample_error: Optional[str] = None

        for turn in turns:
            result = timed_chat(
                base_url=args.base_url,
                model=args.model,
                messages=turn.messages,
                api_key=args.api_key,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                disable_thinking=args.disable_thinking,
            )
            total_e2e_ms += result.e2e_ms or 0.0
            if turn.is_final:
                final_raw = result.raw_response
                final_prompt_tokens = result.prompt_tokens
            if result.error:
                sample_error = result.error

            turn_row = {
                "sample_id": sample_id,
                "question_id": entry.get("question_id"),
                "question_type": entry.get("question_type"),
                "turn_index": turn.index,
                "is_warmup": turn.is_warmup,
                "is_final": turn.is_final,
                "model_name": args.model,
                "ttft_ms": result.ttft_ms,
                "e2e_ms": result.e2e_ms,
                "decode_ms": result.decode_ms,
                "tpot_ms": result.tpot_ms,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "status": "error" if result.error else "ok",
                "error": result.error,
            }
            append_jsonl(turns_path, turn_row)
            all_turn_rows.append(turn_row)

            if result.error and args.stop_on_error:
                break

        hypothesis_row = {
            "question_id": entry.get("question_id"),
            "hypothesis": final_raw,
        }
        append_jsonl(hypotheses_path, hypothesis_row)

        judge_label, judge_raw, judge_error = judge_response(
            base_url=args.judge_base_url,
            model=args.judge_model,
            api_key=args.judge_api_key,
            question_type=str(entry.get("question_type")),
            question_id=str(entry.get("question_id")),
            question=str(entry.get("question", "")),
            answer=str(entry.get("answer", "")),
            hypothesis=final_raw,
            timeout=args.timeout,
        )
        correct = float(judge_label) if judge_label is not None else None
        if args.judge_model:
            eval_row = {
                "question_id": entry.get("question_id"),
                "hypothesis": final_raw,
                "autoeval_label": {
                    "model": args.judge_model,
                    "label": judge_label,
                },
                "judge_response": judge_raw,
                "judge_error": judge_error,
            }
            append_jsonl(eval_results_path, eval_row)

        sample_row = {
            "sample_id": sample_id,
            "question_id": entry.get("question_id"),
            "question_type": entry.get("question_type"),
            "model_name": args.model,
            "num_turns": len(turns),
            "total_e2e_ms": total_e2e_ms,
            "final_prompt_tokens": final_prompt_tokens,
            "question": entry.get("question"),
            "gold_answer": entry.get("answer"),
            "raw_response": final_raw,
            "correct": correct,
            "score_method": "official_longmemeval_judge" if args.judge_model else "not_scored_write_hypotheses_for_official_eval",
            "judge_model": args.judge_model,
            "judge_response": judge_raw,
            "judge_error": judge_error,
            "error": sample_error,
        }
        append_jsonl(samples_path, sample_row)
        all_sample_rows.append(sample_row)

        running = summarize(all_turn_rows, all_sample_rows)
        print(
            f"[{i}/{len(rows)}] sid={sample_id} turns={len(turns)} "
            f"final_tokens={final_prompt_tokens} correct={correct} "
            f"running_acc={running['accuracy']}",
            flush=True,
        )

    summary = summarize(all_turn_rows, all_sample_rows)
    summary.update({
        "dataset": str(args.dataset),
        "turns_path": str(turns_path),
        "samples_path": str(samples_path),
        "hypotheses_path": str(hypotheses_path),
        "eval_results_path": str(eval_results_path) if args.judge_model else None,
    })
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {json.dumps(summary, ensure_ascii=False)}", flush=True)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LongMemEval replay-mode TTFT and final-answer benchmark")
    parser.add_argument("--dataset", required=True, help="LongMemEval .json or .jsonl path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL, e.g. http://host:17100/v1")
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument("--api-key", default=os.environ.get("API_KEY", "EMPTY"))
    parser.add_argument("--topk-context", type=int, default=40, help="latest N sessions to replay")
    parser.add_argument("--history-format", choices=["json", "nl"], default="json")
    parser.add_argument("--user-only", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--judge-model", default=None, help="optional judge model; enables official LongMemEval yes/no evaluation")
    parser.add_argument("--judge-base-url", default=None, help="judge OpenAI-compatible base URL")
    parser.add_argument("--judge-api-key", default=os.environ.get("JUDGE_API_KEY", os.environ.get("API_KEY", "EMPTY")))
    return parser


def main() -> None:
    parser = build_arg_parser()
    try:
        run(parser.parse_args())
    except KeyboardInterrupt:
        print("\n[abort] interrupted", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
