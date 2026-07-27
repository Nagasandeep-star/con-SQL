"""Unit tests for the independent query-complexity estimator."""

import unittest

from src.qce import QueryComplexityEstimator
from src.qce.complexity_features import ComplexityFeatureExtractor


class QueryComplexityEstimatorTests(unittest.TestCase):
    def setUp(self):
        self.estimator = QueryComplexityEstimator()

    def test_simple_query_uses_smallest_budget(self):
        result = self.estimator.estimate("List all students")

        self.assertEqual("Simple", result.complexity_level)
        self.assertEqual("Top-3", result.retrieval_budget)

    def test_medium_query_uses_medium_budget(self):
        result = self.estimator.estimate("Count the orders after 2020 for each customer")

        self.assertEqual("Medium", result.complexity_level)
        self.assertEqual("Top-5", result.retrieval_budget)

    def test_complex_query_uses_largest_budget(self):
        result = self.estimator.estimate(
            "Find the top customers whose total orders are greater than the average "
            "for each region and rank them by the highest total"
        )

        self.assertEqual("Complex", result.complexity_level)
        self.assertEqual("Top-8", result.retrieval_budget)

    def test_features_depend_only_on_query_text(self):
        features = ComplexityFeatureExtractor().extract("Show unique customers before 2020")

        self.assertEqual(5, features.token_count)
        self.assertEqual(1, features.distinct_hint_count)
        self.assertEqual(1, features.comparison_count)

    def test_empty_query_is_rejected(self):
        with self.assertRaises(ValueError):
            self.estimator.estimate("   ")


if __name__ == "__main__":
    unittest.main()
