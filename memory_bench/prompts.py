"""按题型构造标准 system prompt。

复刻 AgentLongBench 的 prompt 模板, 但本项目自包含, 不依赖原仓库。
"""
from __future__ import annotations

from .types import HistoryLabel, KnowledgeLabel, QuestionType


def build_system_prompt(
    qt: QuestionType,
    knowledge_label: str,
    history_label: str,
) -> str:
    """根据 (题型, 知识, 历史风格) 三元组返回标准 system prompt。"""
    if knowledge_label == KnowledgeLabel.KNOWLEDGE_FREE.value:
        return _build_masked_prompt(qt, history_label)
    return _build_pokemon_prompt(qt, history_label)


def _build_pokemon_prompt(qt: QuestionType, history_label: str) -> str:
    if qt in {QuestionType.COUNT_FREQUENCY_TOOL, QuestionType.COUNT_CORRECTNESS_ENV}:
        return (
            "You are analyzing a guess-the-Pokemon dialogue. Full conversation history "
            "(including tool results and feedback) is provided. "
            "Answer the question based on the tool return values or environment feedback. "
            "Wrap your answer in <answer></answer>. "
            "If the answer is a number, answer in arabic numerals (e.g., 3 not three)."
        )
    if qt is QuestionType.COUNT_FREQUENCY_ENV:
        return (
            "You are analyzing a guess-the-Pokemon dialogue. Full conversation history with feedback is provided.\n"
            "Answer the question by counting occurrences of a property value across all rounds' feedback.\n"
            "Wrap your final answer (a number) in <answer></answer>."
        )
    if qt is QuestionType.FIND_ROUND_LARGEST_VALUE_ENV:
        return (
            "You are analyzing a guess-the-Pokemon dialogue. Full conversation history with feedback is provided.\n"
            "Answer the question by identifying which round has the highest total base stats.\n"
            "Wrap your final answer (round number) in <answer></answer>."
        )
    if qt is QuestionType.WEIGHTED_SUMMATION_ENV:
        return (
            "You are analyzing a guess-the-Pokemon dialogue. Full conversation history with feedback is provided.\n"
            "Calculate the weighted scores for two rounds using this weighted rule:\n"
            "- Type: 6 points per correct item\n"
            "- Ability: 5 points per correct item\n"
            "- Base Stats: 4 points per correct item\n"
            "- Evolution: 3 points per correct item\n"
            "- Generation: 2 points per correct item\n"
            "- Other sections: 1 point per correct item\n"
            "Then compute the absolute difference between the two rounds' scores.\n"
            "Wrap your final answer (difference value) in <answer></answer>."
        )
    if qt is QuestionType.FIND_DUPLICATES_TOOL:
        return (
            "You are analyzing a guess-the-Pokemon dialogue. Full conversation history is provided.\n"
            "Answer the question with yes/no or true/false based on whether a Pokemon "
            "appears in both tool results.\n"
            "Wrap your final answer in <answer></answer>."
        )
    if qt is QuestionType.FIND_TARGET_OFFSETS_TOOL:
        return (
            "You are analyzing a guess-the-Pokemon dialogue. Full conversation history is provided.\n"
            "Answer the question by identifying the two Pokemon names in order.\n"
            "Format your answer as: <answer>Pokemon1 and Pokemon2</answer>"
        )
    if qt is QuestionType.INTERSECTION and history_label == HistoryLabel.VERBOSE.value:
        return (
            "You are reviewing a guess-the-Pokemon dialogue. Full history messages "
            "(including tool results) are provided; infer the intersection list for the "
            "target round's tool call. "
            "Each round is defined as: user guess -> optional tool call -> feedback. "
            "The first round has no tool call; the first tool call appears after the user's "
            "second guess (called round 2), and so on. "
            "Return only the intersection as a comma-separated list or JSON array. Do not "
            "call any tools. Wrap the final list in <answer></answer>."
        )
    if qt is QuestionType.INTERSECTION and history_label == HistoryLabel.CONCISE.value:
        return (
            "You are an expert analyst for a deductive reasoning game. "
            "The full conversation history with system feedback is provided.\n"
            "Your task is to analyze the logical progression and constraints revealed "
            "throughout the dialogue to deduce the hidden target Pokemon.\n"
            "The correct answer must be logically consistent with the entire history of feedback.\n"
            "Return only the Pokemon name. Do not call any tools. "
            "Wrap your final answer in <answer></answer>."
        )
    return (
        "You are analyzing a guess-the-Pokemon dialogue. Full conversation history is provided. "
        "Answer the question and wrap your final answer in <answer></answer>."
    )


def _build_masked_prompt(qt: QuestionType, history_label: str) -> str:
    if qt in {QuestionType.COUNT_FREQUENCY_TOOL, QuestionType.COUNT_CORRECTNESS_ENV}:
        return (
            "You are analyzing a masked guess-the-entity dialogue. Full conversation history "
            "(including tool results and feedback) is provided. "
            "Answer the question based on the tool return values or environment feedback. "
            "Wrap your final answer in <answer></answer>."
        )
    if qt is QuestionType.COUNT_FREQUENCY_ENV:
        return (
            "You are analyzing a masked guess-the-entity dialogue. Full conversation history with feedback is provided.\n"
            "Answer the question by counting occurrences of a property value across all rounds' feedback.\n"
            "Wrap your final answer (a number) in <answer></answer>."
        )
    if qt is QuestionType.FIND_ROUND_LARGEST_VALUE_ENV:
        return (
            "You are analyzing a masked guess-the-entity dialogue. Full conversation history with feedback is provided.\n"
            "Answer the question by identifying which round has the highest attr_2 total (numeric field).\n"
            "Wrap your final answer (round number) in <answer></answer>."
        )
    if qt is QuestionType.WEIGHTED_SUMMATION_ENV:
        return (
            "You are analyzing a masked guess-the-entity dialogue. Full conversation history with feedback is provided.\n"
            "Calculate the weighted scores for two rounds using this weighted rule:\n"
            "- attr_1: 6 points per correct item\n"
            "- attr_4: 5 points per correct item\n"
            "- attr_2: 4 points per correct item\n"
            "- attr_5: 3 points per correct item\n"
            "- attr_3: 2 points per correct item\n"
            "- attr_6: 1 point per correct item\n"
            "Then compute the absolute difference between the two rounds' scores.\n"
            "Wrap your final answer (difference value) in <answer></answer>."
        )
    if qt is QuestionType.FIND_DUPLICATES_TOOL:
        return (
            "You are analyzing a masked guess-the-entity dialogue. Full conversation history is provided.\n"
            "Answer the question with yes/no or true/false based on whether an entity id "
            "appears in both tool results.\n"
            "Wrap your final answer in <answer></answer>."
        )
    if qt is QuestionType.FIND_TARGET_OFFSETS_TOOL:
        return (
            "You are analyzing a masked guess-the-entity dialogue. Full conversation history is provided.\n"
            "Answer the question by identifying the two entity ids in order.\n"
            "Format your answer as: <answer>id1 and id2</answer>"
        )
    if qt is QuestionType.INTERSECTION and history_label == HistoryLabel.VERBOSE.value:
        return (
            "You are reviewing a masked guess-the-entity dialogue. Full history messages "
            "(including tool results) are provided; infer the intersection list for the "
            "target round's tool call. "
            "Return only the intersection as a comma-separated list or JSON array. "
            "Wrap the final list in <answer></answer>."
        )
    if qt is QuestionType.INTERSECTION and history_label == HistoryLabel.CONCISE.value:
        return (
            "You are an expert analyst for a deductive reasoning game with masked ids. "
            "The full conversation history with system feedback is provided.\n"
            "Analyze the constraints to deduce the hidden target id.\n"
            "Return only the masked id. Wrap your final answer in <answer></answer>."
        )
    return (
        "You are analyzing a masked guess-the-entity dialogue. Full conversation history is provided. "
        "Answer the question and wrap your final answer in <answer></answer>."
    )
