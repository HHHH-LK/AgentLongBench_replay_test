"""数据结构和枚举的单一定义点。

所有跨模块共享的类型都集中在这里, 避免循环引用和重复定义。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ════════════════════════════════════════════════════════════════════════
# 枚举类
# ════════════════════════════════════════════════════════════════════════

class Role(str, Enum):
    """OpenAI Chat Completions 标准消息角色。"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Backend(str, Enum):
    """推理后端类型。本项目支持 OpenAI 兼容 HTTP 服务 (vLLM serve)。"""
    API = "api"


class TurnStatus(str, Enum):
    """单轮请求的执行状态。"""
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"


class QuestionType(str, Enum):
    """AgentLongBench 8 种题型 (slug 形式, 与文件路径对应)。"""
    # Tool Response 类
    COUNT_FREQUENCY_TOOL = "count_frequency_tool"
    FIND_DUPLICATES_TOOL = "find_duplicates_tool"
    FIND_TARGET_OFFSETS_TOOL = "find_target_offsets_tool"
    # Env Response 类
    COUNT_CORRECTNESS_ENV = "count_correctness_env"
    COUNT_FREQUENCY_ENV = "count_frequency_env"
    FIND_ROUND_LARGEST_VALUE_ENV = "find_round_largest_value_env"
    WEIGHTED_SUMMATION_ENV = "weighted_summation_env"
    # Final Guess
    INTERSECTION = "intersection"

    @property
    def paper_label(self) -> str:
        """返回数据集里 question_type 字段的真实字符串 (带空格和括号)。"""
        return _PAPER_LABEL[self]

    @classmethod
    def from_any(cls, raw: str) -> "QuestionType":
        """从论文标签或 slug 反解析。"""
        if raw is None:
            raise ValueError("question_type is None")
        # 直接匹配 slug
        for m in cls:
            if m.value == raw:
                return m
        # 反向匹配论文标签
        for m, label in _PAPER_LABEL.items():
            if label == raw:
                return m
        # 容错: 大小写无关 + 去掉非字母数字
        norm = "".join(ch for ch in raw.lower() if ch.isalnum())
        for m in cls:
            if "".join(ch for ch in m.value if ch.isalnum()) == norm:
                return m
        raise ValueError(f"unknown question_type: {raw!r}")


_PAPER_LABEL = {
    QuestionType.COUNT_FREQUENCY_TOOL: "Count Frequency(Tool)",
    QuestionType.FIND_DUPLICATES_TOOL: "Find Duplicates(Tool)",
    QuestionType.FIND_TARGET_OFFSETS_TOOL: "Find Target Offsets(Tool)",
    QuestionType.COUNT_CORRECTNESS_ENV: "Count Correctness(Env)",
    QuestionType.COUNT_FREQUENCY_ENV: "Count Frequency(Env)",
    QuestionType.FIND_ROUND_LARGEST_VALUE_ENV: "Find Round with Largest Value(Env)",
    QuestionType.WEIGHTED_SUMMATION_ENV: "Weighted Summation(Env)",
    QuestionType.INTERSECTION: "Intersection",
}


class KnowledgeLabel(str, Enum):
    """数据集知识标签 (路径里的 ki/kf 部分)。"""
    KNOWLEDGE_INTENSIVE = "knowledge_intensive"   # 真 Pokemon 名
    KNOWLEDGE_FREE = "knowledge_free"             # 符号化 attr_x


class HistoryLabel(str, Enum):
    """数据集历史风格 (路径里的 c/v 部分)。"""
    CONCISE = "Concise-Response"
    VERBOSE = "Verbose-Response"


class ParseKind(str, Enum):
    """parse_response 解析出的结果类型。"""
    NUMBER = "number"
    BOOLEAN = "boolean"
    LIST = "list"
    INTERSECTION_LIST = "intersection_list"
    FINAL_ANSWER = "final_answer"
    UNKNOWN = "unknown"


# ════════════════════════════════════════════════════════════════════════
# 数据类
# ════════════════════════════════════════════════════════════════════════

@dataclass
class Turn:
    """切分层的输出: 一轮请求的不可变快照。

    runner 拿到 Turn 后只读, 用 request_messages 发请求, 不修改 Turn 本身。
    """
    id: str                                   # "{sample_id}-t{index}"
    sample_id: str
    index: int                                # 0-based
    request_messages: List[Dict[str, Any]]    # 累积 prefix (含 system)
    is_final: bool = False                    # 最后一轮 (问 question 那一轮)
    is_warmup: bool = False                   # warmup 轮 (一般 index==0)

    def message_count(self) -> int:
        return len(self.request_messages)


@dataclass
class TurnMetrics:
    """单轮请求的测量结果。落盘到 turns.jsonl, 一轮一行。"""
    # 关联
    turn_id: str
    sample_id: str
    turn_index: int

    # 后端信息
    backend: str
    model_name: str

    # 标志位 (从 Turn 透传, 便于分析时不用 join)
    is_warmup: bool = False
    is_final: bool = False

    # 计时 (毫秒)
    ttft_ms: Optional[float] = None           # Time-To-First-Token
    e2e_ms: Optional[float] = None            # End-To-End
    decode_ms: Optional[float] = None         # = e2e - ttft
    tpot_ms: Optional[float] = None           # Time-Per-Output-Token

    # token 计数 (从 usage 字段拿)
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cumulative_prompt_tokens: Optional[int] = None

    # 仅 final 轮保留
    raw_response: Optional[str] = None

    # 状态
    status: str = TurnStatus.OK.value
    error: Optional[str] = None

    # 优化对照预留 (基线版本全部为 None)
    compression_time_ms: Optional[float] = None
    compression_ratio: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SampleResult:
    """单条样本的最终结果 (含准确率信息)。落盘到 samples.jsonl。"""
    sample_id: str
    backend: str
    model_name: str
    question_type: str
    knowledge_label: Optional[str]
    history_label: Optional[str]

    num_turns: int
    total_e2e_ms: float
    final_prompt_tokens: Optional[int]

    raw_response: Optional[str]               # 末轮模型完整输出
    pred_answer: Any = None                   # parse_response 解析结果
    parse_kind: Optional[str] = None
    gold_answer: Any = None                   # 数据集 answer 字段
    correct: Optional[float] = None           # 0/1 或 F1 分数 (题型相关)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
