"""切分层: 把一条 sample 切成 N 个累积 prefix 快照。

不变量 (必须满足):
  1. 严格前缀: turns[k].request_messages[:len(turns[k-1])] == turns[k-1]
  2. 单调增长: len(turns[k]) > len(turns[k-1])
  3. 仅最后一轮 is_final == True
  4. 第 0 轮 is_warmup == True
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

from .types import Role, Turn


def split_into_turns(
    sample: Dict[str, Any],
    system_prompt: str,
    *,
    max_turns: Optional[int] = None,
) -> List[Turn]:
    sid = str(sample.get("id"))
    history: List[Dict[str, Any]] = [
        {"role": Role.SYSTEM.value, "content": system_prompt}
    ]
    pending: List[Dict[str, Any]] = []
    turns: List[Turn] = []

    def emit(is_final: bool) -> None:
        idx = len(turns)
        turns.append(Turn(
            id=f"{sid}-t{idx}",
            sample_id=sid,
            index=idx,
            request_messages=deepcopy(history),
            is_final=is_final,
            is_warmup=(idx == 0),
        ))

    for msg in sample.get("messages") or []:
        role = msg.get("role")
        if role == Role.SYSTEM.value:
            continue                             # 跳过原 system, 用我们重写的
        if role == Role.USER.value:
            for buf in pending:                  # flush 缓冲区
                history.append(deepcopy(buf))
            pending.clear()
            history.append(deepcopy(msg))
            emit(is_final=False)
        else:
            pending.append(msg)                  # asst/tool 攒进缓冲区

    # 收尾: flush + 追加 question
    for buf in pending:
        history.append(deepcopy(buf))
    pending.clear()

    question = sample.get("question")
    if question is not None:
        history.append({"role": Role.USER.value, "content": question})
        emit(is_final=True)
    elif turns:
        # 数据集没有显式 question 字段, 把最后一轮强行标 final
        turns[-1] = Turn(
            id=turns[-1].id,
            sample_id=turns[-1].sample_id,
            index=turns[-1].index,
            request_messages=turns[-1].request_messages,
            is_final=True,
            is_warmup=turns[-1].is_warmup,
        )

    # 限制最大轮数 (保留首轮 warmup + 末轮 final)
    if max_turns is not None and len(turns) > max_turns:
        kept = turns[: max_turns - 1] + [turns[-1]]
        for i, t in enumerate(kept):
            t.index = i
            t.id = f"{sid}-t{i}"
            t.is_warmup = (i == 0)
        turns = kept

    return turns


# ════════════════════════════════════════════════════════════════════════
# 自检 (运行 `python -m memory_bench.replay` 触发)
# ════════════════════════════════════════════════════════════════════════

def _selftest() -> None:
    sample = {
        "id": "demo",
        "messages": [
            {"role": "system", "content": "old_sys"},
            {"role": "user", "content": "猜"},
            {"role": "assistant", "content": "皮卡丘?"},
            {"role": "user", "content": "不对"},
            {"role": "assistant", "content": "调用工具"},
            {"role": "tool", "content": "[结果]"},
            {"role": "assistant", "content": "是雷丘?"},
            {"role": "user", "content": "也不对"},
        ],
        "question": "你猜了几次?",
    }
    turns = split_into_turns(sample, "new_sys")

    assert len(turns) == 4, f"应有 4 轮, 实际 {len(turns)}"
    assert turns[0].is_warmup
    assert turns[-1].is_final
    assert sum(t.is_final for t in turns) == 1, "仅最后一轮应为 final"
    for i in range(1, len(turns)):
        prev = turns[i - 1].request_messages
        cur = turns[i].request_messages
        assert cur[: len(prev)] == prev, f"Turn {i} 不是 Turn {i-1} 的前缀"
        assert len(cur) > len(prev), f"Turn {i} 长度未增长"
    assert turns[0].request_messages[0]["content"] == "new_sys", "system 未替换"
    assert turns[-1].request_messages[-1]["content"] == "你猜了几次?"
    print(f"[OK] selftest passed, {len(turns)} turns generated")
    for t in turns:
        print(f"  Turn {t.index}  is_final={t.is_final}  msgs={t.message_count()}")


if __name__ == "__main__":
    _selftest()
