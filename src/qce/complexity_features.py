"""Feature extraction for natural-language query complexity estimation."""

import re
from dataclasses import dataclass
from typing import FrozenSet


@dataclass(frozen=True)
class ComplexityFeatures:
    """Lexical and structural signals extracted from a natural-language query."""

    token_count: int
    aggregation_count: int
    comparison_count: int
    conjunction_count: int
    join_hint_count: int
    nesting_hint_count: int
    sorting_or_limit_count: int
    distinct_hint_count: int


class ComplexityFeatureExtractor:
    """Extract deterministic, dependency-free complexity features from text."""

    AGGREGATION_TERMS: FrozenSet[str] = frozenset(
        {"average", "avg", "count", "total", "sum", "minimum", "maximum", "min", "max"}
    )
    COMPARISON_TERMS: FrozenSet[str] = frozenset(
        {"before", "after", "between", "greater", "less", "more", "fewer", "over", "under", "than"}
    )
    CONJUNCTION_TERMS: FrozenSet[str] = frozenset({"and", "or", "but", "with", "without"})
    JOIN_HINT_TERMS: FrozenSet[str] = frozenset(
        {"each", "associated", "related", "belonging", "per", "among", "from"}
    )
    NESTING_HINT_TERMS: FrozenSet[str] = frozenset(
        {"whose", "that", "which", "where", "exists", "any", "all", "every"}
    )
    SORTING_OR_LIMIT_TERMS: FrozenSet[str] = frozenset(
        {"top", "bottom", "highest", "lowest", "first", "last", "latest", "earliest", "order", "rank"}
    )
    DISTINCT_TERMS: FrozenSet[str] = frozenset({"distinct", "unique", "different"})

    def extract(self, query: str) -> ComplexityFeatures:
        """Return normalized lexical features for ``query``.

        The extractor intentionally uses only the query text. It does not
        inspect database schema, retrieved examples, or generated SQL.
        """
        tokens = re.findall(r"[a-z0-9_]+", query.lower())
        token_set = set(tokens)

        return ComplexityFeatures(
            token_count=len(tokens),
            aggregation_count=len(token_set & self.AGGREGATION_TERMS),
            comparison_count=len(token_set & self.COMPARISON_TERMS),
            conjunction_count=sum(token in self.CONJUNCTION_TERMS for token in tokens),
            join_hint_count=len(token_set & self.JOIN_HINT_TERMS),
            nesting_hint_count=len(token_set & self.NESTING_HINT_TERMS),
            sorting_or_limit_count=len(token_set & self.SORTING_OR_LIMIT_TERMS),
            distinct_hint_count=len(token_set & self.DISTINCT_TERMS),
        )
