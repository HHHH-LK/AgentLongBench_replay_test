"""分析层: 把 turns.jsonl + samples.jsonl 聚成 P50/P90/P99 + 准确率。

输出三个文件 (CSV):
  turn_summary.csv     — 按 turn_index 聚合性能指标
  bucket_summary.csv   — 按 prompt_tokens 分桶聚合
  accuracy.csv         — 单样本对错明细 + 总分
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def percentile(vals: List[Optional[float]], q: float) -> Optional[float]:
    clean = sorted(v for v in vals if v is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    k = (len(clean) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(clean) - 1)
    return clean[lo] if lo == hi else clean[lo] + (clean[hi] - clean[lo]) * (k - lo)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ════════════════════════════════════════════════════════════════════════
# 1. 按 turn_index 聚合
# ════════════════════════════════════════════════════════════════════════

def summarize_by_turn(turns_path: Path, out_csv: Path, drop_warmup: bool = True) -> None:
    rows = _load_jsonl(turns_path)
    by_turn: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if drop_warmup and r.get("is_warmup"):
            continue
        if r.get("status") != "ok":
            continue
        by_turn[int(r["turn_index"])].append(r)

    metric_keys = ["ttft_ms", "decode_ms", "tpot_ms", "e2e_ms"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["turn_index", "n", "avg_prompt_tokens"]
        for m in metric_keys:
            header += [f"{m}_p50", f"{m}_p90", f"{m}_p99"]
        w.writerow(header)
        for k in sorted(by_turn):
            bucket = by_turn[k]
            tokens = [r.get("prompt_tokens") for r in bucket if r.get("prompt_tokens")]
            avg_t = int(statistics.mean(tokens)) if tokens else 0
            row = [k, len(bucket), avg_t]
            for m in metric_keys:
                vals = [r.get(m) for r in bucket]
                p50 = percentile(vals, 0.5)
                p90 = percentile(vals, 0.9)
                p99 = percentile(vals, 0.99)
                row += [
                    f"{p50:.2f}" if p50 is not None else "",
                    f"{p90:.2f}" if p90 is not None else "",
                    f"{p99:.2f}" if p99 is not None else "",
                ]
            w.writerow(row)
    print(f"[turn] -> {out_csv}")


# ════════════════════════════════════════════════════════════════════════
# 2. 按 prompt_tokens 分桶 (看 TTFT 随上下文长度的曲线)
# ════════════════════════════════════════════════════════════════════════

_BUCKETS = [(0, 4_000), (4_000, 16_000), (16_000, 32_000),
            (32_000, 64_000), (64_000, 128_000), (128_000, 10_000_000)]


def _bucket_label(n: int) -> str:
    for lo, hi in _BUCKETS:
        if lo <= n < hi:
            if hi >= 10_000_000:
                return f">={lo//1000}k"
            return f"{lo//1000}k-{hi//1000}k"
    return "?"


def summarize_by_bucket(turns_path: Path, out_csv: Path, drop_warmup: bool = True) -> None:
    rows = _load_jsonl(turns_path)
    by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if drop_warmup and r.get("is_warmup"):
            continue
        if r.get("status") != "ok":
            continue
        if not r.get("prompt_tokens"):
            continue
        by_bucket[_bucket_label(r["prompt_tokens"])].append(r)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prompt_tokens_bucket", "n",
                    "ttft_p50", "ttft_p90", "ttft_p99",
                    "tpot_p50", "tpot_p90", "tpot_p99"])
        # 按 bucket 顺序写
        order = ["0k-4k", "4k-16k", "16k-32k", "32k-64k", "64k-128k", ">=128k"]
        for label in order:
            bucket = by_bucket.get(label) or []
            if not bucket:
                continue
            ttft = [r.get("ttft_ms") for r in bucket]
            tpot = [r.get("tpot_ms") for r in bucket]
            row = [label, len(bucket)]
            for vals in (ttft, tpot):
                for q in (0.5, 0.9, 0.99):
                    v = percentile(vals, q)
                    row.append(f"{v:.2f}" if v is not None else "")
            w.writerow(row)
    print(f"[bucket] -> {out_csv}")


# ════════════════════════════════════════════════════════════════════════
# 3. 准确率
# ════════════════════════════════════════════════════════════════════════

def summarize_accuracy(samples_path: Path, out_csv: Path) -> Dict[str, Any]:
    rows = _load_jsonl(samples_path)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    correct = 0.0
    per_qtype: Dict[str, List[float]] = defaultdict(list)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "sample_id", "question_type", "knowledge_label", "history_label",
            "pred_answer", "gold_answer", "correct",
        ])
        for r in rows:
            c = r.get("correct")
            w.writerow([
                r.get("sample_id"),
                r.get("question_type"),
                r.get("knowledge_label"),
                r.get("history_label"),
                r.get("pred_answer"),
                r.get("gold_answer"),
                c,
            ])
            if c is not None:
                total += 1
                correct += c
                per_qtype[r.get("question_type", "unknown")].append(c)

    overall = (correct / total) if total else None
    print(f"[acc] -> {out_csv}")
    print(f"[acc] overall: {correct:.1f} / {total} = {overall}")
    for qt, scores in per_qtype.items():
        avg = sum(scores) / len(scores)
        print(f"      {qt}: {sum(scores):.1f} / {len(scores)} = {avg:.4f}")

    return {"total": total, "correct": correct, "overall": overall}


# ════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(description="聚合 baseline 测试结果")
    p.add_argument("--run-dir", type=Path, required=True,
                   help="包含 turns.jsonl / samples.jsonl 的目录")
    p.add_argument("--keep-warmup", action="store_true",
                   help="保留 warmup 轮 (默认剔除, 避免冷启动失真)")
    args = p.parse_args()

    turns_path = args.run_dir / "turns.jsonl"
    samples_path = args.run_dir / "samples.jsonl"

    drop = not args.keep_warmup
    if turns_path.exists():
        summarize_by_turn(turns_path, args.run_dir / "turn_summary.csv", drop_warmup=drop)
        summarize_by_bucket(turns_path, args.run_dir / "bucket_summary.csv", drop_warmup=drop)
    else:
        print(f"[warn] {turns_path} not found")
    if samples_path.exists():
        summarize_accuracy(samples_path, args.run_dir / "accuracy.csv")
    else:
        print(f"[warn] {samples_path} not found")


if __name__ == "__main__":
    main()
