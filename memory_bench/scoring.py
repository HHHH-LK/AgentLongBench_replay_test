"""按题型对单条样本打分。

复刻 AgentLongBench 的评分语义:
  - 数值/布尔/单名 → 0/1 accuracy
  - INTERSECTION + Verbose → set F1
  - FIND_TARGET_OFFSETS_TOOL → 顺序敏感, 1.0 / 0.5 / 0.0
"""
from __future__ import annotations

import re
from typing import Any, List, Optional

from .types import HistoryLabel, KnowledgeLabel, QuestionType


# ════════════════════════════════════════════════════════════════════════
# 规范化工具
# ════════════════════════════════════════════════════════════════════════

def _to_number(val: Any) -> Optional[int]:
    if val is None:
        return None
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, (int, float)):
        return int(val)
    m = re.search(r"-?\d+(?:\.\d+)?", str(val))
    if not m:
        return None
    try:
        return int(float(m.group(0)))
    except ValueError:
        return None


def _to_boolean(val: Any) -> Optional[bool]:
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return val > 0
    text = str(val).lower().strip()
    if any(w in text for w in ("yes", "true", "1")):
        return True
    if any(w in text for w in ("no", "false", "0")):
        return False
    return None


def _normalize_name(name: str, *, knowledge_label: str) -> str:
    if not name:
        return ""
    norm = str(name).strip().lower()
    chars = " -'\"."
    if knowledge_label == KnowledgeLabel.KNOWLEDGE_FREE.value:
        chars = " -_'\"."
    for ch in chars:
        norm = norm.replace(ch, "")
    return norm


def _to_pair_list(val: Any, *, knowledge_label: str) -> Optional[List[str]]:
    if val is None:
        return None
    if isinstance(val, list):
        return [_normalize_name(x, knowledge_label=knowledge_label) for x in val if isinstance(x, str)]
    if isinstance(val, str):
        text = val.strip("[](){}")
        parts = re.split(r"[,;]|\s+and\s+", text, flags=re.IGNORECASE)
        if len(parts) >= 2:
            return [_normalize_name(p.strip(), knowledge_label=knowledge_label) for p in parts]
    return None


def _to_set(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(v).strip() for v in val if str(v).strip()]
    if isinstance(val, str):
        return [v.strip() for chunk in val.replace("\n", ",").split(",")
                for v in chunk.split() if v.strip()]
    return []


# ════════════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════════════

def score_one(
    qt: QuestionType,
    history_label: str,
    knowledge_label: str,
    pred_answer: Any,
    gold_answer: Any,
) -> Optional[float]:
    """单条样本的打分 → 0.0 / 1.0 (个别题型支持小数, 如 0.5 / F1)。

    返回 None 表示题型不支持或 gold 缺失, 不计入聚合。
    """
    if qt in {
        QuestionType.COUNT_FREQUENCY_TOOL,
        QuestionType.COUNT_CORRECTNESS_ENV,
        QuestionType.COUNT_FREQUENCY_ENV,
        QuestionType.FIND_ROUND_LARGEST_VALUE_ENV,
        QuestionType.WEIGHTED_SUMMATION_ENV,
    }:
        g = _to_number(gold_answer)
        p = _to_number(pred_answer)
        if g is None:
            return None
        return 1.0 if p == g else 0.0

    if qt is QuestionType.FIND_DUPLICATES_TOOL:
        g = _to_boolean(gold_answer)
        p = _to_boolean(pred_answer)
        if g is None:
            return None
        return 1.0 if p == g else 0.0

    if qt is QuestionType.FIND_TARGET_OFFSETS_TOOL:
        g = _to_pair_list(gold_answer, knowledge_label=knowledge_label)
        p = _to_pair_list(pred_answer, knowledge_label=knowledge_label)
        if not g:
            return None
        if not p:
            return 0.0
        if len(p) >= 2 and len(g) >= 2 and p[0] == g[0] and p[1] == g[1]:
            return 1.0
        if len(p) >= 1 and len(g) >= 2 and p[0] == g[0]:
            return 0.5
        return 0.0

    if qt is QuestionType.INTERSECTION and history_label == HistoryLabel.VERBOSE.value:
        # set F1
        g_set = set(_to_set(gold_answer))
        p_set = set(_to_set(pred_answer))
        if not g_set and not p_set:
            return 1.0
        if not g_set or not p_set:
            return 0.0
        inter = len(g_set & p_set)
        prec = inter / len(p_set)
        rec = inter / len(g_set)
        return (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    if qt is QuestionType.INTERSECTION and history_label == HistoryLabel.CONCISE.value:
        g_norm = _normalize_name(str(gold_answer or ""), knowledge_label=knowledge_label)
        p_norm = _normalize_name(str(pred_answer or ""), knowledge_label=knowledge_label)
        if not g_norm:
            return None
        return 1.0 if p_norm == g_norm else 0.0

    return None
