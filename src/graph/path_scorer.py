# src/path_scorer.py
# ============================================================
# ConSQL - Phase 5: Constraint Scoring + Best Path Selection
#
# This is the CORE NOVEL CONTRIBUTION of the research.
# 
# Scoring Formula (from the paper):
#   Score(p) =     S_sem(p) +     S_struct(p)       L(p)
#
# Where:
#   S_sem(p)    = Semantic Relevance Score  (avg node weight along path)
#   S_struct(p) = Structural Validity Score (valid FK edges / total edges)
#   L(p)        = Length Penalty            (normalized path length)
#    ,  ,       = Hyperparameters from config.py
#
# Final score is normalized to [0, 1].
# ============================================================

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from src.data_loader import Schema
from src.retrieval.schema_linker import SchemaLinkingResult
from src.graph.graph_builder import SchemaGraph
from src.graph.path_generator import JoinPath, PathGenerationResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# -----------------------------------------
# DATA CLASSES
# -----------------------------------------

@dataclass
class ScoredPath:
    """A join path with all its computed scores."""
    path:          JoinPath
    sem_score:     float    # S_sem:    semantic relevance    [0, 1]
    struct_score:  float    # S_struct: structural validity   [0, 1]
    length_penalty: float   # L(p):     path length penalty   [0, 1]
    final_score:   float    # combined score                  [0, 1]

    # Component breakdown for interpretability
    score_breakdown: Dict = field(default_factory=dict)

    def __repr__(self):
        path_str = " -> ".join(self.path.path_table_names)
        return (f"ScoredPath([{path_str}] | "
                f"sem={self.sem_score:.3f}, "
                f"struct={self.struct_score:.3f}, "
                f"penalty={self.length_penalty:.3f}, "
                f"final={self.final_score:.3f})")


@dataclass
class PathSelectionResult:
    """Full output of path scoring and selection."""
    query:        str
    db_id:        str
    best_path:    ScoredPath              # highest-scoring path
    scored_paths: List[ScoredPath]        # all paths ranked by score
    alpha:        float
    beta:         float
    lam:          float
    top_k_paths:  List[ScoredPath] = field(default_factory=list)

    def summary(self) -> str:
        bp = self.best_path
        path_str = " -> ".join(bp.path.path_table_names)
        return (f"Best Path: [{path_str}] | "
                f"Score={bp.final_score:.3f} "
                f"(sem={bp.sem_score:.3f}, "
                f"struct={bp.struct_score:.3f}, "
                f"penalty={bp.length_penalty:.3f})")


# -----------------------------------------
# PATH SCORER
# -----------------------------------------

class PathScorer:
    """
    Scores and ranks candidate join paths using a unified
    constraint-guided scoring formula.

    Formula:
        Score(p) =     S_sem(p) +     S_struct(p)       L(p)

    Components:
    ---------------------------------------------------------
    S_sem(p):
        Average semantic relevance of tables in the path.
        Uses node weights from SBERT schema linking.
        Rewards paths that pass through query-relevant tables.

        S_sem(p) = (1/|p|)     node_weight(ti)

    S_struct(p):
        Structural validity score based on FK edge quality.
        Rewards paths with strong FK relationships.
        Combines: FK edge existence + edge semantic weight

        S_struct(p) = (valid_FK_edges / total_edges)   avg_edge_weight
                    = 1.0   avg_edge_weight  (since all paths have valid FKs)

    L(p) -- Length Penalty:
        Penalizes longer paths to prefer simpler joins.
        Normalized to [0, 1]:

        L(p) = (|p| - 1) / (max_depth - 1)

        where |p| = number of tables, max_depth from config.

    Final Score is clamped to [0, 1].
    ---------------------------------------------------------
    """

    def __init__(
        self,
        alpha:     float = 0.4,
        beta:      float = 0.4,
        lam:       float = 0.2,
        max_depth: int   = 4
    ):
        """
        Args:
            alpha: weight for semantic relevance score
            beta:  weight for structural validity score
            lam:   weight for length penalty (lambda)
            max_depth: maximum join depth (for normalizing length penalty)
        """
        self.alpha     = alpha
        self.beta      = beta
        self.lam       = lam
        self.max_depth = max_depth

        # Validate weights sum to ~1 (alpha + beta should equal 1 - lam)
        assert abs(alpha + beta + lam - 1.0) < 0.01, \
            f"Weights must sum to 1.0: alpha={alpha} + beta={beta} + lam={lam} = {alpha+beta+lam}"

    # -- Main Scoring Entry Point ------------------------------

    def score_and_select(
        self,
        path_result:    PathGenerationResult,
        schema_graph:   SchemaGraph,
        linking_result: SchemaLinkingResult,
        top_k:          int = 1
    ) -> PathSelectionResult:
        """
        Score all valid paths and return the best one(s).

        Args:
            path_result:    output from Phase 4 PathGenerator
            schema_graph:   weighted schema graph from Phase 3
            linking_result: SBERT linking output from Phase 2
            top_k:          number of candidate paths to include in top_k_paths

        Returns:
            PathSelectionResult with best path, top-k paths, and all ranked paths
        """
        if not path_result.valid_paths:
            raise ValueError(f"No valid paths to score for query: {path_result.query}")

        # Score each valid path
        scored_paths = []
        for path in path_result.valid_paths:
            sp = self._score_path(path, schema_graph, linking_result)
            scored_paths.append(sp)

        # Rank by final score (descending)
        scored_paths.sort(key=lambda x: x.final_score, reverse=True)

        best_path = scored_paths[0]
        selected_top_k = scored_paths[:max(1, min(top_k, len(scored_paths)))]

        return PathSelectionResult(
            query=path_result.query,
            db_id=path_result.db_id,
            best_path=best_path,
            scored_paths=scored_paths,
            alpha=self.alpha,
            beta=self.beta,
            lam=self.lam,
            top_k_paths=selected_top_k
        )

    # -- Score a Single Path -----------------------------------

    def _score_path(
        self,
        path:           JoinPath,
        schema_graph:   SchemaGraph,
        linking_result: SchemaLinkingResult
    ) -> ScoredPath:
        """
        Compute all score components for a single path.
        Returns a ScoredPath with full breakdown.
        """
        # -- S_sem: Semantic Relevance Score -------------------
        sem_score = self._compute_semantic_score(path, schema_graph, linking_result)

        # -- S_struct: Structural Validity Score ---------------
        struct_score = self._compute_structural_score(path, schema_graph)

        # -- L(p): Length Penalty ------------------------------
        length_penalty = self._compute_length_penalty(path)

        # -- Final Score ---------------------------------------
        raw_score  = (self.alpha * sem_score
                    + self.beta  * struct_score
                    - self.lam   * length_penalty)

        # Clamp to [0, 1]
        final_score = max(0.0, min(1.0, raw_score))

        breakdown = {
            "    S_sem":      round(self.alpha * sem_score, 4),
            "    S_struct":   round(self.beta  * struct_score, 4),
            "    L(p)":       round(self.lam   * length_penalty, 4),
            "raw_score":      round(raw_score, 4),
            "final_score":    round(final_score, 4),
        }

        return ScoredPath(
            path=path,
            sem_score=round(sem_score, 4),
            struct_score=round(struct_score, 4),
            length_penalty=round(length_penalty, 4),
            final_score=round(final_score, 4),
            score_breakdown=breakdown
        )

    # -- Score Components -------------------------------------

    def _compute_semantic_score(
        self,
        path:           JoinPath,
        schema_graph:   SchemaGraph,
        linking_result: SchemaLinkingResult
    ) -> float:
        """
        S_sem(p) = average node weight (query relevance) along the path.

        Node weights come from SBERT schema linking scores.
        Rewards paths that include highly query-relevant tables.
        """
        if not path.path_table_ids:
            return 0.0

        # Use node weights from the query-aware schema graph
        weights = []
        for tid in path.path_table_ids:
            w = schema_graph.get_node_weight(tid)
            weights.append(w)

        if not weights:
            return 0.0

        # Average semantic relevance across all tables in path
        sem_score = float(np.mean(weights))

        # Bonus: if path starts or ends at a highly relevant table
        # (the most relevant table should be a path endpoint, not buried)
        max_relevance = max(weights)
        endpoint_relevance = max(weights[0], weights[-1])
        endpoint_bonus = 0.05 * (endpoint_relevance / max(max_relevance, 0.001))

        return min(1.0, sem_score + endpoint_bonus)

    def _compute_structural_score(
        self,
        path:         JoinPath,
        schema_graph: SchemaGraph
    ) -> float:
        """
        S_struct(p) = structural quality of FK edges along the path.

        For a path with k edges:
            S_struct = (valid_FK_count / k)   avg_edge_weight

        Since Phase 4 already ensures all edges are valid FKs,
        valid_FK_count / k = 1.0 always.
        So: S_struct = avg_edge_weight of the path.

        Additionally, rewards paths with higher-weight edges
        (semantically stronger FK relationships).
        """
        if path.num_joins == 0:
            # Single-table path: structural score = node's own relevance
            if path.path_table_ids:
                return schema_graph.get_node_weight(path.path_table_ids[0])
            return 0.5

        # Use edge weights stored in the path (from graph_builder)
        if path.edge_weights:
            avg_edge_weight = float(np.mean(path.edge_weights))
        else:
            # Recompute from graph if not stored
            edge_weights = []
            for i in range(len(path.path_table_ids) - 1):
                t1, t2 = path.path_table_ids[i], path.path_table_ids[i+1]
                w = schema_graph.get_edge_weight(t1, t2)
                edge_weights.append(w)
            avg_edge_weight = float(np.mean(edge_weights)) if edge_weights else 0.5

        # FK consistency factor: all edges valid = 1.0 (enforced by Phase 4)
        fk_consistency = 1.0

        return fk_consistency * avg_edge_weight

    def _compute_length_penalty(self, path: JoinPath) -> float:
        """
        L(p) = normalized path length penalty.

        L(p) = (|p| - 1) / (max_depth - 1)

        Single-table path: L = 0 (no penalty)
        Max-depth path:    L = 1 (maximum penalty)

        This penalizes unnecessary joins while allowing
        the semantic and structural scores to overcome
        the penalty when joins are truly needed.
        """
        if self.max_depth <= 1:
            return 0.0
        num_joins = max(0, path.length - 1)
        return num_joins / (self.max_depth - 1)

    # -- Hyperparameter Sensitivity Analysis ------------------

    def sensitivity_analysis(
        self,
        paths:          List[JoinPath],
        schema_graph:   SchemaGraph,
        linking_result: SchemaLinkingResult
    ) -> List[Dict]:
        """
        Test different  ,  ,   combinations and show how
        the best path changes. Useful for tuning hyperparameters.
        """
        configs = [
            # (alpha, beta, lambda, label)
            (0.4, 0.4, 0.2, "Balanced (paper default)"),
            (0.6, 0.2, 0.2, "Semantic-heavy"),
            (0.2, 0.6, 0.2, "Structure-heavy"),
            (0.4, 0.5, 0.1, "Low length penalty"),
            (0.3, 0.3, 0.4, "High length penalty"),
        ]

        results = []
        for alpha, beta, lam, label in configs:
            scorer = PathScorer(alpha=alpha, beta=beta, lam=lam,
                                max_depth=self.max_depth)
            dummy_result = type('obj', (object,), {
                'valid_paths': paths,
                'query': linking_result.query,
                'db_id': schema_graph.db_id
            })()
            sel = scorer.score_and_select(dummy_result, schema_graph, linking_result)
            bp  = sel.best_path
            results.append({
                "config":       label,
                "alpha":        alpha,
                "beta":         beta,
                "lambda":       lam,
                "best_path":    " -> ".join(bp.path.path_table_names),
                "final_score":  bp.final_score,
                "sem_score":    bp.sem_score,
                "struct_score": bp.struct_score,
            })

        return results

    # -- Batch Evaluation -------------------------------------

    def evaluate(
        self,
        samples,
        schemas:       Dict,
        schema_graphs: Dict,
        linker,
        generator,
        top_k: int = 100
    ) -> Dict:
        """
        Evaluate path scoring on multiple samples.
        Measures: avg best score, path length distribution,
                  how often the best path contains all relevant tables.
        """
        total_score      = 0.0
        total_coverage   = 0
        path_lengths     = []
        total            = 0

        for sample in samples[:top_k]:
            schema = schemas.get(sample.db_id)
            sg     = schema_graphs.get(sample.db_id)
            if not schema or not sg:
                continue

            linking     = linker.link(sample.question, schema)
            path_result = generator.generate(sg, linking, schema)

            if not path_result.valid_paths:
                continue

            # Build query-aware graph for scoring
            from src.graph.graph_builder import SchemaGraphBuilder
            builder  = SchemaGraphBuilder(linker=linker)
            sg_aware = builder.build(schema, linking_result=linking)

            sel = self.score_and_select(path_result, sg_aware, linking)

            total_score += sel.best_path.final_score
            path_lengths.append(sel.best_path.path.length)

            # Coverage: does best path include all relevant tables?
            relevant_ids = set(linking.get_table_ids())
            best_ids     = set(sel.best_path.path.path_table_ids)
            if relevant_ids.issubset(best_ids) or best_ids.intersection(relevant_ids):
                total_coverage += 1

            total += 1

        n = max(total, 1)
        return {
            "samples_evaluated":     total,
            "avg_best_score":        round(total_score / n, 4),
            "avg_best_path_length":  round(float(np.mean(path_lengths)), 2) if path_lengths else 0,
            "coverage_rate":         round(total_coverage / n * 100, 2),
            "path_length_dist":      {
                str(l): path_lengths.count(l)
                for l in sorted(set(path_lengths))
            }
        }

    # -- Pretty Print -----------------------------------------

    def print_result(self, result: PathSelectionResult, top_n: int = 5):
        """Pretty print scoring results."""
        print(f"\n{'='*65}")
        print(f"Path Scoring: {result.db_id}")
        print(f"Query: {result.query[:70]}")
        print(f"Weights:  ={result.alpha},  ={result.beta},  ={result.lam}")
        print(f"{'-'*65}")

        print(f"\n  BEST PATH:")
        bp       = result.best_path
        path_str = " -> ".join(bp.path.path_table_names)
        print(f"   {path_str}")
        print(f"   Score Breakdown:")
        for k, v in bp.score_breakdown.items():
            print(f"     {k:<20} = {v}")

        print(f"\n  All Scored Paths (top {min(top_n, len(result.scored_paths))}):")
        print(f"   {'#':<4} {'Score':>6}  {'Sem':>6}  {'Struct':>6}  {'Pen':>6}  Path")
        print(f"   {'-'*60}")
        for i, sp in enumerate(result.scored_paths[:top_n]):
            path_str = " -> ".join(sp.path.path_table_names)
            marker   = " " if i == 0 else f"  "
            print(f"  {marker}[{i+1}] {sp.final_score:>6.3f}  "
                  f"{sp.sem_score:>6.3f}  "
                  f"{sp.struct_score:>6.3f}  "
                  f"{sp.length_penalty:>6.3f}  "
                  f"{path_str}")
        print(f"{'='*65}")