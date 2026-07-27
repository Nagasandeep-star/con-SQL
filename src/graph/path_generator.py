# src/path_generator.py
# ============================================================
# ConSQL - Phase 4: Candidate Path Generator + Invalid Path Removal
# 
# Given a schema graph and relevant tables from SBERT linking,
# this module:
#   1. Generates ALL candidate join paths via DFS/BFS
#   2. Removes INVALID paths (cycles, no FK edge, disconnected)
#   3. Returns only structurally VALID candidate paths
#
# A path P = [t1, t2, ..., tk] where each (ti, ti+1) is a
# valid FK edge in the schema graph.
# ============================================================

import logging
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque

import networkx as nx

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from src.data_loader import Schema, Table
from src.retrieval.schema_linker import SchemaLinkingResult, TableScore
from src.graph.graph_builder import SchemaGraph

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────

@dataclass
class JoinPath:
    """
    A single candidate join path through the schema graph.
    path_table_ids: ordered list of table IDs in the join
    path_table_names: corresponding table names (for readability)
    edge_weights: FK edge weights along the path
    is_valid: True if passed all structural constraint checks
    rejection_reason: why it was rejected (if invalid)
    """
    path_table_ids:   List[int]
    path_table_names: List[str]
    edge_weights:     List[float] = field(default_factory=list)
    is_valid:         bool = True
    rejection_reason: str = ""

    @property
    def length(self) -> int:
        """Number of tables in the path."""
        return len(self.path_table_ids)

    @property
    def num_joins(self) -> int:
        """Number of JOIN operations (tables - 1)."""
        return max(0, self.length - 1)

    @property
    def avg_edge_weight(self) -> float:
        if not self.edge_weights:
            return 0.0
        return sum(self.edge_weights) / len(self.edge_weights)

    def __repr__(self):
        path_str = " → ".join(self.path_table_names)
        return f"JoinPath([{path_str}], valid={self.is_valid})"

    def __hash__(self):
        return hash(tuple(self.path_table_ids))

    def __eq__(self, other):
        return self.path_table_ids == other.path_table_ids


@dataclass
class PathGenerationResult:
    """
    Full output of path generation for one query.
    Contains both valid and invalid paths for analysis.
    """
    query: str
    db_id: str
    all_paths:     List[JoinPath]    # all generated paths
    valid_paths:   List[JoinPath]    # paths that passed all checks
    invalid_paths: List[JoinPath]    # paths that were rejected

    @property
    def num_valid(self) -> int:
        return len(self.valid_paths)

    @property
    def num_invalid(self) -> int:
        return len(self.invalid_paths)

    def summary(self) -> str:
        return (f"PathGen({self.db_id}) | "
                f"Total: {len(self.all_paths)} | "
                f"Valid: {self.num_valid} | "
                f"Invalid: {self.num_invalid}")


# ─────────────────────────────────────────
# PATH GENERATOR
# ─────────────────────────────────────────

class PathGenerator:
    """
    Generates candidate join paths from a schema graph.

    Strategy:
    - Start from each relevant table (from SBERT linking)
    - Use DFS to explore paths up to max_depth
    - Collect all simple paths (no repeated nodes)
    - Apply structural validity filters to remove invalid paths

    Structural Constraints Enforced:
    ✅ C1: All consecutive table pairs must have a valid FK edge
    ✅ C2: No cycles — each table appears at most once in a path
    ✅ C3: Path length must be within [1, max_depth]
    ✅ C4: Path must contain at least one relevant table (from linking)
    ✅ C5: All relevant tables must be reachable/included (coverage check)
    ✅ C6: No redundant paths (duplicate table sequences)
    """

    def __init__(
        self,
        max_depth: int = 4,
        max_paths: int = 50,
        require_all_relevant: bool = False
    ):
        """
        Args:
            max_depth: maximum number of tables in a join path (d_max)
            max_paths: cap on candidate paths returned per query
            require_all_relevant: if True, only keep paths that include
                                   ALL relevant tables (strict mode)
        """
        self.max_depth           = max_depth
        self.max_paths           = max_paths
        self.require_all_relevant = require_all_relevant

    # ── Main Entry Point ─────────────────────────────────────

    def generate(
        self,
        schema_graph: SchemaGraph,
        linking_result: SchemaLinkingResult,
        schema: Schema
    ) -> PathGenerationResult:
        """
        Generate and filter candidate join paths.

        Args:
            schema_graph:   weighted schema graph (from Phase 3)
            linking_result: SBERT schema linking output (from Phase 2)
            schema:         original Schema object

        Returns:
            PathGenerationResult with valid and invalid paths
        """
        query          = linking_result.query
        db_id          = schema.db_id
        relevant_ids   = set(linking_result.get_table_ids())
        graph          = schema_graph.graph

        # ── Step 1: Generate all candidate paths via DFS ──────
        raw_paths = self._generate_raw_paths(
            graph, schema_graph, relevant_ids
        )

        # ── Step 2: Apply structural validity filters ─────────
        valid_paths, invalid_paths = self._filter_paths(
            raw_paths, graph, schema_graph, relevant_ids
        )

        # ── Step 3: Fallback if no valid paths ────────────────
        if not valid_paths:
            valid_paths = self._fallback(
                graph, schema_graph, relevant_ids
            )
            logger.warning(
                f"⚠️  No valid paths found for '{query[:50]}'. "
                f"Fallback produced {len(valid_paths)} path(s)."
            )

        # ── Step 4: Cap at max_paths ──────────────────────────
        valid_paths = valid_paths[:self.max_paths]

        all_paths = valid_paths + invalid_paths
        return PathGenerationResult(
            query=query,
            db_id=db_id,
            all_paths=all_paths,
            valid_paths=valid_paths,
            invalid_paths=invalid_paths
        )

    # ── Raw Path Generation (DFS) ────────────────────────────

    def _generate_raw_paths(
        self,
        graph: nx.Graph,
        sg: SchemaGraph,
        relevant_ids: Set[int]
    ) -> List[JoinPath]:
        """
        Use DFS from each relevant table to enumerate all simple paths
        up to max_depth. Returns raw (unfiltered) path list.
        """
        all_paths: List[JoinPath] = []
        seen_path_keys: Set[tuple] = set()

        # Start DFS from each relevant table
        start_nodes = list(relevant_ids) if relevant_ids else list(graph.nodes())

        for start_id in start_nodes:
            if start_id not in graph:
                continue
            paths_from_start = self._dfs(graph, sg, start_id, seen_path_keys)
            all_paths.extend(paths_from_start)

        return all_paths

    def _dfs(
        self,
        graph: nx.Graph,
        sg: SchemaGraph,
        start_id: int,
        seen_keys: Set[tuple]
    ) -> List[JoinPath]:
        """
        Depth-First Search from start_id.
        Returns all simple paths of length 1..max_depth.
        """
        results: List[JoinPath] = []

        # Stack stores: (current_path_ids, current_edge_weights, visited_set)
        stack = [
            ([start_id], [], {start_id})
        ]

        while stack:
            path_ids, edge_weights, visited = stack.pop()
            current_node = path_ids[-1]

            # Record this path (even single-table paths are valid for simple queries)
            path_key = tuple(path_ids)
            if path_key not in seen_keys:
                seen_keys.add(path_key)
                path_names = [sg.table_id_to_name.get(tid, str(tid)) for tid in path_ids]
                results.append(JoinPath(
                    path_table_ids=list(path_ids),
                    path_table_names=path_names,
                    edge_weights=list(edge_weights)
                ))

            # Stop expanding if at max depth
            if len(path_ids) >= self.max_depth:
                continue

            # Expand to neighbors
            for neighbor in graph.neighbors(current_node):
                if neighbor not in visited:
                    new_path    = path_ids + [neighbor]
                    edge_w      = graph[current_node][neighbor].get("weight", 1.0)
                    new_weights = edge_weights + [edge_w]
                    new_visited = visited | {neighbor}
                    stack.append((new_path, new_weights, new_visited))

        return results

    # ── Structural Validity Filters ──────────────────────────

    def _filter_paths(
        self,
        paths: List[JoinPath],
        graph: nx.Graph,
        sg: SchemaGraph,
        relevant_ids: Set[int]
    ) -> Tuple[List[JoinPath], List[JoinPath]]:
        """
        Apply all structural constraints.
        Returns (valid_paths, invalid_paths).
        """
        valid   = []
        invalid = []

        for path in paths:
            ok, reason = self._check_constraints(path, graph, sg, relevant_ids)
            if ok:
                path.is_valid = True
                valid.append(path)
            else:
                path.is_valid         = False
                path.rejection_reason = reason
                invalid.append(path)

        return valid, invalid

    def _check_constraints(
        self,
        path: JoinPath,
        graph: nx.Graph,
        sg: SchemaGraph,
        relevant_ids: Set[int]
    ) -> Tuple[bool, str]:
        """
        Check all structural constraints for a single path.
        Returns (is_valid, rejection_reason).

        Constraints:
        C1: All edges must be valid FK edges
        C2: No cycles (DFS ensures this, but double-check)
        C3: Path length within bounds
        C4: Must include at least one relevant table
        C5: (Optional) Must include all relevant tables
        """
        ids = path.path_table_ids

        # C3: Length constraint
        if len(ids) == 0:
            return False, "C3: Empty path"
        if len(ids) > self.max_depth:
            return False, f"C3: Path too long ({len(ids)} > {self.max_depth})"

        # C2: No repeated tables (no cycles)
        if len(ids) != len(set(ids)):
            return False, "C2: Cycle detected (repeated table)"

        # C1: All consecutive pairs must have a valid FK edge
        for i in range(len(ids) - 1):
            t1, t2 = ids[i], ids[i+1]
            if not graph.has_edge(t1, t2):
                return False, f"C1: No FK edge between {sg.table_id_to_name.get(t1)} and {sg.table_id_to_name.get(t2)}"

        # C4: Must contain at least one relevant table
        path_set = set(ids)
        if relevant_ids and not path_set.intersection(relevant_ids):
            return False, "C4: No relevant table in path"

        # C5: Optional — must contain ALL relevant tables
        if self.require_all_relevant and relevant_ids:
            missing = relevant_ids - path_set
            if missing:
                missing_names = [sg.table_id_to_name.get(m, str(m)) for m in missing]
                return False, f"C5: Missing relevant tables: {missing_names}"

        return True, ""

    # ── Fallback ─────────────────────────────────────────────

    def _fallback(
        self,
        graph: nx.Graph,
        sg: SchemaGraph,
        relevant_ids: Set[int]
    ) -> List[JoinPath]:
        """
        If no valid paths found (e.g. relevant tables are disconnected),
        try with relaxed depth (max_depth + 2) or return single-table paths.
        """
        # Try single-table paths for each relevant table
        fallback_paths = []
        for tid in relevant_ids:
            if tid in graph:
                name = sg.table_id_to_name.get(tid, str(tid))
                fallback_paths.append(JoinPath(
                    path_table_ids=[tid],
                    path_table_names=[name],
                    edge_weights=[],
                    is_valid=True,
                    rejection_reason=""
                ))

        # Also try NetworkX shortest paths between relevant table pairs
        rel_list = list(relevant_ids)
        for i in range(len(rel_list)):
            for j in range(i+1, len(rel_list)):
                t1, t2 = rel_list[i], rel_list[j]
                if t1 in graph and t2 in graph:
                    try:
                        sp = nx.shortest_path(graph, t1, t2, weight=None)
                        if len(sp) <= self.max_depth + 2:
                            names   = [sg.table_id_to_name.get(n, str(n)) for n in sp]
                            weights = [graph[sp[k]][sp[k+1]].get("weight", 1.0)
                                       for k in range(len(sp)-1)]
                            fallback_paths.append(JoinPath(
                                path_table_ids=list(sp),
                                path_table_names=names,
                                edge_weights=weights,
                                is_valid=True,
                                rejection_reason=""
                            ))
                    except nx.NetworkXNoPath:
                        pass

        return fallback_paths[:self.max_paths]

    # ── Utilities ─────────────────────────────────────────────

    def print_result(
        self,
        result: PathGenerationResult,
        show_invalid: bool = False,
        top_n: int = 10
    ):
        """Pretty print path generation results."""
        print(f"\n{'='*65}")
        print(f"Path Generation: {result.db_id}")
        print(f"Query: {result.query[:70]}")
        print(f"{'-'*65}")
        print(f"Total Generated : {len(result.all_paths)}")
        print(f"✅ Valid Paths  : {result.num_valid}")
        print(f"❌ Invalid Paths: {result.num_invalid}")

        print(f"\n✅ Valid Paths (top {min(top_n, result.num_valid)}):")
        for i, path in enumerate(result.valid_paths[:top_n]):
            path_str = " → ".join(path.path_table_names)
            w_str    = f"avg_w={path.avg_edge_weight:.3f}" if path.edge_weights else "single-table"
            print(f"  [{i+1}] {path_str}  ({w_str})")

        if show_invalid and result.invalid_paths:
            print(f"\n❌ Invalid Paths (sample):")
            for path in result.invalid_paths[:5]:
                path_str = " → ".join(path.path_table_names)
                print(f"  ✗ {path_str}  | Reason: {path.rejection_reason}")

        print(f"{'='*65}")

    def evaluate(
        self,
        samples,
        schemas: Dict,
        schema_graphs: Dict,
        linker,
        top_k: int = 100
    ) -> Dict:
        """
        Evaluate path generation on a set of samples.
        Measures: avg valid paths, avg invalid paths, path coverage.
        """
        total_valid    = 0
        total_invalid  = 0
        total_coverage = 0
        total          = 0

        for sample in samples[:top_k]:
            schema = schemas.get(sample.db_id)
            sg     = schema_graphs.get(sample.db_id)
            if not schema or not sg:
                continue

            linking = linker.link(sample.question, schema)
            result  = self.generate(sg, linking, schema)

            total_valid   += result.num_valid
            total_invalid += result.num_invalid
            total         += 1

            # Coverage: do valid paths cover all relevant tables?
            relevant_ids = set(linking.get_table_ids())
            covered_ids  = set()
            for path in result.valid_paths:
                covered_ids.update(path.path_table_ids)
            coverage = len(covered_ids & relevant_ids) / max(len(relevant_ids), 1)
            total_coverage += coverage

        n = max(total, 1)
        return {
            "samples_evaluated":    total,
            "avg_valid_paths":      round(total_valid   / n, 2),
            "avg_invalid_paths":    round(total_invalid / n, 2),
            "avg_table_coverage":   round(total_coverage / n * 100, 2),
        }