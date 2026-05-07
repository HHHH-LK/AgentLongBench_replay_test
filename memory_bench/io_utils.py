"""JSONL 读写 + AgentLongBench 数据集路径推断。"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """写一行并立即 flush, 避免长跑丢数据。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


# ════════════════════════════════════════════════════════════════════════
# AgentLongBench 路径约定
#   .../ki-c/32k/final_guess/intersection.jsonl
#         │ │
#         │ └─ history label:  c=Concise-Response  v=Verbose-Response
#         └─── knowledge label: ki=knowledge_intensive  kf=knowledge_free
# ════════════════════════════════════════════════════════════════════════

_KNOWLEDGE_MAP = {"ki": "knowledge_intensive", "kf": "knowledge_free"}
_HISTORY_MAP = {"c": "Concise-Response", "v": "Verbose-Response"}


def infer_context_from_path(path: Path) -> Tuple[str, str]:
    """从路径推断 (knowledge_label, history_label)。

    路径里必须含一段形如 ``ki-c`` 或 ``kf-v`` 的目录名。
    找不到时回退到 ``knowledge_intensive`` + ``Concise-Response``。
    """
    for part in (p.lower() for p in path.parts):
        m = re.fullmatch(r"(ki|kf)-(c|v)", part)
        if m:
            return _KNOWLEDGE_MAP[m.group(1)], _HISTORY_MAP[m.group(2)]
    return "knowledge_intensive", "Concise-Response"


def require_single_question_type(rows: Iterable[Dict[str, Any]]) -> str:
    types = {r.get("question_type") for r in rows if r.get("question_type")}
    if not types:
        raise ValueError("dataset has no question_type field")
    if len(types) > 1:
        raise ValueError(f"dataset has multiple question_types: {sorted(types)}")
    return next(iter(types))
