# src/graph_builder.py
# ============================================================
# ConSQL - Phase 3: Schema Graph Builder
# Builds a weighted graph from database schema where:
#   Nodes = Tables
#   Edges = Foreign Key relationships (weighted by similarity)
# ============================================================

import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from src.data_loader import Schema, Table, Column, ForeignKey
from src.retrieval.schema_linker import SBERTSchemaLinker, SchemaLinkingResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────

@dataclass
class EdgeInfo:
    """Metadata for a foreign key edge in the schema graph."""
    from_table_id: int
    to_table_id: int
    from_table_name: str
    to_table_name: str
    from_col_name: str
    to_col_name: str
    weight: float           # semantic similarity between connected tables

    def __repr__(self):
        return (f"Edge({self.from_table_name}.{self.from_col_name} "
                f"→ {self.to_table_name}.{self.to_col_name}, w={self.weight:.3f})")


@dataclass
class SchemaGraph:
    """
    A weighted undirected graph representing a database schema.
    Nodes = table_ids, Edges = FK relationships.
    Includes query-aware node weights from SBERT schema linking.
    """
    db_id: str
    graph: nx.Graph = field(default_factory=nx.Graph)
    edge_info: Dict[Tuple[int, int], List[EdgeInfo]] = field(default_factory=dict)
    node_weights: Dict[int, float] = field(default_factory=dict)   # table_id → relevance score
    table_id_to_name: Dict[int, str] = field(default_factory=dict)
    table_name_to_id: Dict[str, int] = field(default_factory=dict)

    def get_neighbors(self, table_id: int) -> List[int]:
        """Return adjacent table IDs (connected via FK)."""
        if table_id not in self.graph:
            return []
        return list(self.graph.neighbors(table_id))

    def get_edge_weight(self, t1_id: int, t2_id: int) -> float:
        """Return edge weight between two tables."""
        if self.graph.has_edge(t1_id, t2_id):
            return self.graph[t1_id][t2_id].get("weight", 1.0)
        return 0.0

    def get_node_weight(self, table_id: int) -> float:
        """Return query relevance score for a table node."""
        return self.node_weights.get(table_id, 0.0)

    def has_path(self, t1_id: int, t2_id: int) -> bool:
        """Check if two tables are connected (directly or indirectly)."""
        return nx.has_path(self.graph, t1_id, t2_id)

    def summary(self) -> str:
        return (f"SchemaGraph({self.db_id}) | "
                f"Nodes: {self.graph.number_of_nodes()} | "
                f"Edges: {self.graph.number_of_edges()}")


# ─────────────────────────────────────────
# GRAPH BUILDER
# ─────────────────────────────────────────

class SchemaGraphBuilder:
    """
    Builds a weighted schema graph from a database Schema object.

    The graph is:
      - Undirected (FK relationships are traversable both ways)
      - Weighted edges based on semantic similarity between tables
      - Node weights set from SBERT schema linking scores

    This serves as input to the Path Generator (Phase 4).
    """

    def __init__(self, linker: Optional[SBERTSchemaLinker] = None):
        """
        Args:
            linker: SBERTSchemaLinker instance for computing semantic
                    edge weights. If None, all edges get weight=1.0.
        """
        self.linker = linker
        # Cache table-pair similarities to avoid recomputation
        self._sim_cache: Dict[str, float] = {}

    # ── Main Build ───────────────────────────────────────────

    def build(
        self,
        schema: Schema,
        linking_result: Optional[SchemaLinkingResult] = None
    ) -> SchemaGraph:
        """
        Build a SchemaGraph for a given schema.

        Args:
            schema: The database schema
            linking_result: Optional SBERT result to set node weights

        Returns:
            SchemaGraph with nodes, edges, and weights
        """
        sg = SchemaGraph(db_id=schema.db_id)

        # Step 1: Add only SBERT-selected tables as nodes (candidate set S_r)
        self._add_nodes(sg, schema, linking_result)

        # Step 2: Add FK relationships as edges, but only between tables in S_r
        self._add_edges(sg, schema)

        # Step 3: Log summary
        logger.debug(f"Built {sg.summary()}")
        return sg

    def _add_nodes(
        self,
        sg: SchemaGraph,
        schema: Schema,
        linking_result: Optional[SchemaLinkingResult]
    ):
        """Add table nodes with optional query-relevance weights."""
        # Build lookup for linking scores if available
        link_score_map: Dict[int, float] = {}
        if linking_result:
            for ts in linking_result.all_table_scores:
                link_score_map[ts.table.table_id] = ts.score

        # If linking_result is provided, only include its relevant tables
        selected_ids: Set[int] = set()
        if linking_result:
            selected_ids = set(linking_result.get_table_ids())
            logger.info(f"SBERT candidate tables (S_r): {[t.table.name_original for t in linking_result.relevant_tables]}")
        else:
            # no linking information → include all tables
            selected_ids = set(t.table_id for t in schema.tables)

        for table in schema.tables:
            tid = table.table_id
            if tid not in selected_ids:
                continue

            # Node weight = schema linking score (or 1.0 if not available)
            weight = link_score_map.get(tid, 1.0)

            sg.graph.add_node(
                tid,
                name=table.name,
                name_original=table.name_original,
                weight=weight
            )
            sg.node_weights[tid] = weight
            sg.table_id_to_name[tid] = table.name
            sg.table_name_to_id[table.name] = tid

        logger.info(f"Generated nodes for graph {sg.db_id}: {list(sg.table_id_to_name.values())}")

    def _add_edges(self, sg: SchemaGraph, schema: Schema):
        """
        Add FK edges between tables.
        Edge weight = semantic similarity between the two connected tables
        (computed using SBERT if linker available, else 1.0).
        """
        logger.info(f"Loaded {len(schema.foreign_keys)} foreign keys for schema {schema.db_id}")
        for fk in schema.foreign_keys:
            t1_id = fk.from_table_id
            t2_id = fk.to_table_id

            # Skip self-loops and invalid table IDs
            if t1_id == t2_id or t1_id < 0 or t2_id < 0:
                continue

            # Only create edges when both endpoint tables are present in the candidate set (S_r)
            if not (sg.graph.has_node(t1_id) and sg.graph.has_node(t2_id)):
                # skip edges that connect to nodes not in S_r
                continue

            t1 = schema.get_table_by_id(t1_id)
            t2 = schema.get_table_by_id(t2_id)
            if t1 is None or t2 is None:
                continue

            # Compute edge weight
            weight = self._compute_edge_weight(t1, t2, schema)

            # Get column names for edge metadata
            fc = schema.get_column_by_id(fk.from_col_id)
            tc = schema.get_column_by_id(fk.to_col_id)
            from_col_name = fc.name if fc else "?"
            to_col_name   = tc.name if tc else "?"

            edge_info = EdgeInfo(
                from_table_id=t1_id,
                to_table_id=t2_id,
                from_table_name=t1.name,
                to_table_name=t2.name,
                from_col_name=from_col_name,
                to_col_name=to_col_name,
                weight=weight
            )

            # If edge already exists, keep the maximum semantic weight
            if sg.graph.has_edge(t1_id, t2_id):
                existing_w = sg.graph[t1_id][t2_id].get("weight", 0.0)
                if weight > existing_w:
                    sg.graph[t1_id][t2_id]["weight"] = weight
                    sg.graph[t1_id][t2_id]["fk"] = f"{from_col_name}->{to_col_name}"
                    sg.graph[t1_id][t2_id]["from_col"] = from_col_name
                    sg.graph[t1_id][t2_id]["to_col"] = to_col_name
                key = (min(t1_id, t2_id), max(t1_id, t2_id))
                sg.edge_info.setdefault(key, []).append(edge_info)
            else:
                sg.graph.add_edge(
                    t1_id,
                    t2_id,
                    weight=weight,
                    fk=f"{from_col_name}->{to_col_name}",
                    from_col=from_col_name,
                    to_col=to_col_name,
                    direction=(t1_id, t2_id)
                )
                key = (min(t1_id, t2_id), max(t1_id, t2_id))
                sg.edge_info[key] = [edge_info]
                logger.info(f"Added FK edge: {edge_info}")

        # Summary of edges added
        logger.info(f"Schema graph {sg.db_id}: nodes={sg.graph.number_of_nodes()} edges={sg.graph.number_of_edges()}")

    def _compute_edge_weight(
        self,
        t1: Table,
        t2: Table,
        schema: Schema
    ) -> float:
        """
        Compute semantic similarity between two tables.
        Uses SBERT to encode table names + columns and computes cosine sim.
        Falls back to 1.0 if no linker available.
        """
        if self.linker is None:
            return 1.0

        cache_key = f"{schema.db_id}:{min(t1.table_id, t2.table_id)}:{max(t1.table_id, t2.table_id)}"
        if cache_key in self._sim_cache:
            return self._sim_cache[cache_key]

        # Represent each table as: "table_name col1 col2 col3..."
        def table_text(t: Table) -> str:
            cols = " ".join(c.name for c in t.columns[:5])
            return f"{t.name} {cols}"

        t1_text = table_text(t1)
        t2_text = table_text(t2)

        vecs = self.linker._encode([t1_text, t2_text])
        if len(vecs) < 2:
            return 1.0

        from sklearn.metrics.pairwise import cosine_similarity
        sim = float(cosine_similarity(vecs[0:1], vecs[1:2])[0][0])
        sim = max(0.0, min(1.0, sim))   # clamp to [0,1]

        self._sim_cache[cache_key] = sim
        return sim

    # ── Batch Build ──────────────────────────────────────────

    def build_all(
        self,
        schemas: Dict[str, Schema]
    ) -> Dict[str, SchemaGraph]:
        """Build schema graphs for all schemas (without linking scores)."""
        graphs = {}
        for db_id, schema in schemas.items():
            graphs[db_id] = self.build(schema)
        logger.info(f"✅ Built {len(graphs)} schema graphs.")
        return graphs

    # ── Visualization / Debug ─────────────────────────────────

    def print_graph(self, sg: SchemaGraph, schema: Schema):
        """Pretty print the schema graph."""
        print(f"\n{'='*60}")
        print(f"Schema Graph: {sg.db_id}")
        print(f"{'='*60}")
        print(f"Nodes ({sg.graph.number_of_nodes()}):")
        for node_id in sg.graph.nodes():
            data = sg.graph.nodes[node_id]
            weight = sg.node_weights.get(node_id, 0.0)
            print(f"  [{node_id}] {data.get('name_original', '?'):<25} relevance={weight:.3f}")

        print(f"\nEdges ({sg.graph.number_of_edges()}) [FK Relationships]:")
        for t1_id, t2_id, data in sg.graph.edges(data=True):
            key = (min(t1_id, t2_id), max(t1_id, t2_id))
            infos = sg.edge_info.get(key, [])
            t1_name = sg.table_id_to_name.get(t1_id, str(t1_id))
            t2_name = sg.table_id_to_name.get(t2_id, str(t2_id))
            w = data.get("weight", 0.0)
            print(f"  {t1_name:<20} ↔  {t2_name:<20} weight={w:.3f}")
            for info in infos:
                print(f"    FK: {info.from_col_name} → {info.to_col_name}")

        print(f"\nConnected Components: {nx.number_connected_components(sg.graph)}")
        print(f"{'='*60}")

    def get_graph_stats(self, graphs: Dict[str, SchemaGraph]) -> Dict:
        """Return aggregate statistics across all graphs."""
        num_nodes = [g.graph.number_of_nodes() for g in graphs.values()]
        num_edges = [g.graph.number_of_edges() for g in graphs.values()]
        components = [nx.number_connected_components(g.graph) for g in graphs.values()]
        return {
            "total_graphs":        len(graphs),
            "avg_nodes_per_graph": round(np.mean(num_nodes), 2),
            "avg_edges_per_graph": round(np.mean(num_edges), 2),
            "max_nodes":           int(np.max(num_nodes)),
            "max_edges":           int(np.max(num_edges)),
            "avg_components":      round(np.mean(components), 2),
            "disconnected_graphs": sum(1 for c in components if c > 1),
        }