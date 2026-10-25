import unittest

import pandas as pd

from main import calculate_top_k_counts


class TopKEvaluationTests(unittest.TestCase):
    def test_metrics_do_not_depend_on_answer_key_row_order(self):
        catalogue = pd.DataFrame(
            {
                "service_id": ["S001", "S002", "S003"],
                "service_name": ["Target one", "Target two", "Target three"],
            }
        )
        answer_key = pd.DataFrame(
            {
                "target_id": ["S001", "S002", "S003"],
                "correct_provider_id": ["P001", "P002", "P003"],
            }
        )
        ranked_candidate_ids = [
            ["P001", "P004", "P005"],
            ["P004", "P002", "P005"],
            ["P004", "P005", "P003"],
        ]

        original_order_counts = calculate_top_k_counts(
            catalogue, ranked_candidate_ids, answer_key
        )
        shuffled_order_counts = calculate_top_k_counts(
            catalogue,
            ranked_candidate_ids,
            answer_key.sample(frac=1, random_state=7),
        )

        self.assertEqual(original_order_counts, (1, 3))
        self.assertEqual(shuffled_order_counts, original_order_counts)


if __name__ == "__main__":
    unittest.main()
