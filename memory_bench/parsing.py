"""按题型解析模型 raw_response → 结构化 pred_answer。

复刻 AgentLongBench 的解析逻辑, 但本项目自包含。
"""
from __future__ import annotations

import ast
import re
from typing import Any, List, Optional, Tuple

from .types import HistoryLabel, ParseKind, QuestionType


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL | re.IGNORECASE)


def _extract_answer_tag(text: str) -> Optional[str]:
    if not text:
        return None
    matches = _ANSWER_RE.findall(text)
    return matches[-1].strip() if matches else None


def parse_number(text: str) -> Optional[int]:
    inner = _extract_answer_tag(text)
    if inner is None:
        return None
    matches = re.findall(r"-?\d[\d,]*(?:\.\d+)?", inner)
    for cand in reversed(matches):
        try:
            return int(float(cand.replace(",", "")))
        except ValueError:
            continue
    return None


def parse_boolean(text: str) -> Optional[bool]:
    inner = _extract_answer_tag(text)
    if inner is None:
        return None
    low = inner.lower()
    if re.search(r"\b(no|false|not|doesn't|does not|none|neither)\b", low):
        return False
    if re.search(r"\b(yes|true|contain|contains|appear|appears|does|both)\b", low):
        return True
    m = re.search(r"-?\d[\d,]*", inner)
    if m:
        try:
            return int(float(m.group(0).replace(",", ""))) > 0
        except ValueError:
            return None
    return None


def parse_pair_list(text: str) -> Optional[List[str]]:
    inner = _extract_answer_tag(text)
    if inner is None:
        return None
    inner = inner.strip()
    if inner.startswith("[") and inner.endswith("]"):
        try:
            arr = ast.literal_eval(inner)
            if isinstance(arr, list):
                cleaned = [str(x).strip() for x in arr if str(x).strip()]
                if len(cleaned) >= 2:
                    return cleaned
        except (ValueError, SyntaxError):
            pass
    normalized = re.sub(r"(?i)\band\b", ",", inner)
    normalized = re.sub(r"[\n;|]", ",", normalized)
    tokens: List[str] = []
    for chunk in normalized.split(","):
        item = re.sub(r"^\d+\.?\s*", "", chunk).strip()
        if item:
            tokens.append(item)
    return tokens or None


def parse_intersection_list(text: str) -> List[str]:
    inner = _extract_answer_tag(text)
    if inner is None:
        return []
    inner = inner.strip()
    if inner.startswith("[") and inner.endswith("]"):
        try:
            arr = ast.literal_eval(inner)
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
        except (ValueError, SyntaxError):
            pass
    normalized = re.sub(r"(?i)\band\b", ",", inner)
    normalized = re.sub(r"[\n;|]", ",", normalized)
    return [c.strip() for c in normalized.split(",") if c.strip()]


def parse_final_guess(text: str) -> Optional[str]:
    inner = _extract_answer_tag(text)
    return (inner.strip() if inner else None) or None


def parse_response(
    qt: QuestionType,
    history_label: str,
    raw_text: str,
) -> Tuple[Any, str]:
    """主入口: 按题型分发到具体解析器, 返回 (pred_answer, parse_kind)。"""
    if qt in {
        QuestionType.COUNT_FREQUENCY_TOOL,
        QuestionType.COUNT_CORRECTNESS_ENV,
        QuestionType.COUNT_FREQUENCY_ENV,
        QuestionType.FIND_ROUND_LARGEST_VALUE_ENV,
        QuestionType.WEIGHTED_SUMMATION_ENV,
    }:
        return parse_number(raw_text), ParseKind.NUMBER.value
    if qt is QuestionType.FIND_DUPLICATES_TOOL:
        return parse_boolean(raw_text), ParseKind.BOOLEAN.value
    if qt is QuestionType.FIND_TARGET_OFFSETS_TOOL:
        return parse_pair_list(raw_text), ParseKind.LIST.value
    if qt is QuestionType.INTERSECTION:
        if history_label == HistoryLabel.VERBOSE.value:
            return parse_intersection_list(raw_text), ParseKind.INTERSECTION_LIST.value
        return parse_final_guess(raw_text), ParseKind.FINAL_ANSWER.value
    return None, ParseKind.UNKNOWN.value
