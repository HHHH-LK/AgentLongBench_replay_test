import json
import unittest

from longmemeval_replay.runner import (
    build_replay_turns,
    get_anscheck_prompt,
    load_dataset,
)


class LongMemEvalReplayTests(unittest.TestCase):
    def test_load_dataset_accepts_json_array_and_limit(self):
        path = self.tmp_json([
            {"question_id": "a", "question": "q1", "answer": "a1"},
            {"question_id": "b", "question": "q2", "answer": "a2"},
        ])

        rows = load_dataset(path, offset=1, limit=1)

        self.assertEqual([row["question_id"] for row in rows], ["b"])

    def test_build_replay_turns_appends_sessions_then_final_question(self):
        entry = {
            "question_id": "q1",
            "question_date": "2024-01-03",
            "question": "What degree did I graduate with?",
            "answer": "Business Administration",
            "haystack_dates": ["2024-01-01", "2024-01-02"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "I studied Business Administration."},
                    {"role": "assistant", "content": "Nice."},
                ],
                [
                    {"role": "user", "content": "Unrelated chat."},
                    {"role": "assistant", "content": "Ok."},
                ],
            ],
        }

        turns = build_replay_turns(entry, topk_context=2, history_format="json")

        self.assertEqual(len(turns), 3)
        self.assertFalse(turns[0].is_final)
        self.assertFalse(turns[1].is_final)
        self.assertTrue(turns[2].is_final)
        self.assertIn("History Chats", turns[0].messages[1]["content"])
        self.assertIn("Session 1", turns[0].messages[2]["content"])
        self.assertIn("Session 2", turns[1].messages[3]["content"])
        self.assertIn("Current Date: 2024-01-03", turns[2].messages[-1]["content"])
        self.assertIn("Question: What degree did I graduate with?", turns[2].messages[-1]["content"])

    def test_get_anscheck_prompt_matches_official_task_shape(self):
        prompt = get_anscheck_prompt(
            "single-session-user",
            "What degree did I graduate with?",
            "Business Administration",
            "You graduated with a degree in Business Administration.",
        )
        self.assertIn("Correct Answer: Business Administration", prompt)
        self.assertIn("Model Response: You graduated with a degree", prompt)
        self.assertIn("Answer yes or no only", prompt)

    def test_get_anscheck_prompt_handles_abstention(self):
        prompt = get_anscheck_prompt(
            "single-session-user",
            "What is my password?",
            "The chat history does not provide the password.",
            "I do not have that information.",
            abstention=True,
        )
        self.assertIn("unanswerable question", prompt)
        self.assertIn("Does the model correctly identify", prompt)

    def tmp_json(self, rows):
        import tempfile

        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(rows, f)
        f.close()
        return f.name


if __name__ == "__main__":
    unittest.main()
