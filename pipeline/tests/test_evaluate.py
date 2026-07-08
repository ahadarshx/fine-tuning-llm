import unittest

from evaluate import keyword_recall, rouge_l_f1


class EvaluationMetricTests(unittest.TestCase):
    def test_keyword_recall(self):
        score = keyword_recall(
            "A heuristic never overestimates and preserves optimality.",
            ["never overestimates", "optimal"],
        )
        self.assertEqual(score, 1.0)

    def test_rouge_l_rewards_ordered_overlap(self):
        matching = rouge_l_f1(
            "A star uses path cost and remaining cost.",
            "A star combines path cost with remaining cost.",
        )
        unrelated = rouge_l_f1(
            "Decision trees split examples.",
            "A star combines path cost with remaining cost.",
        )
        self.assertGreater(matching, unrelated)


if __name__ == "__main__":
    unittest.main()
