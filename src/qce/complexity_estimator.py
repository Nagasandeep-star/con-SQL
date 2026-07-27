"""Independent heuristic estimator for natural-language query complexity."""

from src.qce.complexity_features import ComplexityFeatureExtractor, ComplexityFeatures
from src.qce.complexity_result import ComplexityResult


class QueryComplexityEstimator:
    """Estimate Text-to-SQL query complexity using only natural-language text.

    The score is normalized to ``[0.0, 1.0]``. The model is intentionally
    transparent and deterministic so its thresholds can be tuned later from
    evaluation data without coupling it to the inference pipeline.
    """

    SIMPLE_THRESHOLD = 0.30
    MEDIUM_THRESHOLD = 0.60

    def __init__(self, feature_extractor: ComplexityFeatureExtractor | None = None):
        self.feature_extractor = feature_extractor or ComplexityFeatureExtractor()

    def estimate(self, query: str) -> ComplexityResult:
        """Estimate complexity for one natural-language query.

        Args:
            query: The natural-language question to estimate.

        Raises:
            ValueError: If ``query`` is empty or contains only whitespace.
        """
        if not query or not query.strip():
            raise ValueError("Query complexity cannot be estimated from empty text.")

        features = self.feature_extractor.extract(query)
        score = self._score(features)

        if score < self.SIMPLE_THRESHOLD:
            return ComplexityResult(score, "Simple", "Top-3")
        if score < self.MEDIUM_THRESHOLD:
            return ComplexityResult(score, "Medium", "Top-5")
        return ComplexityResult(score, "Complex", "Top-8")

    @staticmethod
    def _score(features: ComplexityFeatures) -> float:
        """Convert independent query-text features into a normalized score."""
        length_signal = min(features.token_count / 25.0, 1.0) * 0.15
        score = (
            length_signal
            + min(features.aggregation_count, 2) * 0.12
            + min(features.comparison_count, 2) * 0.10
            + min(features.conjunction_count, 3) * 0.08
            + min(features.join_hint_count, 2) * 0.10
            + min(features.nesting_hint_count, 2) * 0.12
            + min(features.sorting_or_limit_count, 2) * 0.08
            + min(features.distinct_hint_count, 1) * 0.05
        )
        return round(min(score, 1.0), 3)
