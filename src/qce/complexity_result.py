"""Result type returned by query-complexity estimation."""

from dataclasses import dataclass
from typing import Literal


ComplexityLevel = Literal["Simple", "Medium", "Complex"]
RetrievalBudget = Literal["Top-3", "Top-5", "Top-8"]


@dataclass(frozen=True)
class ComplexityResult:
    """Estimated complexity and the future retrieval budget it suggests."""

    complexity_score: float
    complexity_level: ComplexityLevel
    retrieval_budget: RetrievalBudget

    @property
    def score(self) -> float:
        return self.complexity_score

