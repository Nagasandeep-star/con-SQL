# src/schema_linker.py
# ============================================================
# ConSQL - Phase 2: SBERT Schema Linker
# Maps natural language query -> relevant tables + columns
# using semantic similarity via Sentence-BERT
# ============================================================

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Local imports
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from src.data_loader import Schema, Table, Column, SpiderSample

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# -----------------------------------------
# DATA CLASSES
# -----------------------------------------

@dataclass
class TableScore:
    """A table with its semantic relevance score."""
    table: Table
    score: float
    sim_to_query: float   # raw cosine similarity

    def __repr__(self):
        return f"TableScore(name={self.table.name!r}, score={self.score:.4f})"


@dataclass
class ColumnScore:
    """A column with its semantic relevance score."""
    column: Column
    table: Table
    score: float
    sim_to_query: float

    def __repr__(self):
        return f"ColumnScore(table={self.table.name!r}, col={self.column.name!r}, score={self.score:.4f})"


@dataclass
class SchemaLinkingResult:
    """Full output of schema linking for one query."""
    query: str
    relevant_tables: List[TableScore]        # tables above threshold
    relevant_columns: List[ColumnScore]      # columns above threshold
    all_table_scores: List[TableScore]       # all tables with scores (for debugging)
    all_column_scores: List[ColumnScore]     # all columns with scores

    def get_table_names(self) -> List[str]:
        return [ts.table.name for ts in self.relevant_tables]

    def get_table_ids(self) -> List[int]:
        return [ts.table.table_id for ts in self.relevant_tables]

    def get_columns_for_table(self, table_id: int) -> List[ColumnScore]:
        return [cs for cs in self.relevant_columns if cs.column.table_id == table_id]

    def summary(self) -> str:
        tables = ", ".join(self.get_table_names()) or "None"
        cols = len(self.relevant_columns)
        return f"Query: '{self.query[:60]}...'\n  -> Tables: [{tables}]\n  -> Relevant Columns: {cols}"


# -----------------------------------------
# SCHEMA LINKER
# -----------------------------------------

class SBERTSchemaLinker:
    """
    Uses Sentence-BERT to semantically link a natural language query
    to relevant tables and columns in a database schema.

    Pipeline:
      1. Encode query -> dense vector
      2. Encode each table name + column name -> dense vectors
      3. Compute cosine similarity between query and each schema element
      4. Score =   * table_sim +   * column_sim
      5. Filter elements above threshold  
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.3,
        alpha: float = 0.6,
        beta: float = 0.4,
        device: str = "cpu"
    ):
        """
        Args:
            model_name: HuggingFace SBERT model to use
            similarity_threshold:   -- minimum score to include a schema element
            alpha: weight for table-level similarity
            beta: weight for column-level similarity
            device: 'cpu' or 'cuda'
        """
        self.model_name = model_name
        self.threshold = similarity_threshold
        self.alpha = alpha
        self.beta = beta
        self.device = device

        logger.info(f"Loading SBERT model: {model_name} ...")
        self.model = SentenceTransformer(model_name, device=device)
        logger.info("[OK] SBERT model loaded.")

        # Cache for schema embeddings -- avoid re-encoding same schema
        self._schema_cache: Dict[str, Dict] = {}

    # -- Encoding ---------------------------------------------

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts into dense vectors."""
        if not texts:
            return np.array([])
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True   # L2 norm -> cosine = dot product
        )
        return embeddings

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode the natural language query."""
        return self._encode([query])[0]  # shape: (embedding_dim,)

    def _encode_schema(self, schema: Schema) -> Dict:
        """
        Encode all tables and columns in a schema.
        Results are cached by db_id to avoid repeated computation.

        Returns dict with:
          - table_texts: list of strings used for encoding
          - table_embeddings: np.ndarray (num_tables, dim)
          - column_texts: list of strings
          - column_embeddings: np.ndarray (num_columns, dim)
        """
        if schema.db_id in self._schema_cache:
            return self._schema_cache[schema.db_id]

        # Build table text representations
        # e.g. "table: department columns: department_id, name, creation"
        table_texts = []
        for table in schema.tables:
            col_names = " ".join(c.name for c in table.columns if c.table_id >= 0)
            text = f"table {table.name} contains columns {col_names}"
            table_texts.append(text)

        # Build column text representations
        # e.g. "column: age in table: head"
        column_texts = []
        col_to_table = []   # parallel list: which table does each col belong to
        valid_columns = []  # only columns with valid table_id

        for col in schema.columns:
            if col.table_id < 0:   # skip wildcard column (*)
                continue
            table = schema.get_table_by_id(col.table_id)
            if table is None:
                continue
            text = f"column {col.name} of type {col.col_type} in table {table.name}"
            column_texts.append(text)
            col_to_table.append(table)
            valid_columns.append(col)

        # Encode all
        table_embeddings  = self._encode(table_texts)  if table_texts  else np.array([])
        column_embeddings = self._encode(column_texts) if column_texts else np.array([])

        result = {
            "table_texts":        table_texts,
            "table_embeddings":   table_embeddings,
            "column_texts":       column_texts,
            "column_embeddings":  column_embeddings,
            "valid_columns":      valid_columns,
            "col_to_table":       col_to_table,
        }
        self._schema_cache[schema.db_id] = result
        return result

    # -- Similarity -------------------------------------------

    def _cosine_sim(self, query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query vector and each row in matrix.
        Since embeddings are L2-normalized, cosine sim = dot product.
        Returns array of shape (num_rows,)
        """
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        return cosine_similarity(query_vec.reshape(1, -1), matrix)[0]

    # -- Main Linking -----------------------------------------

    def link(self, query: str, schema: Schema, top_k: Optional[int] = None) -> SchemaLinkingResult:
        """
        Main entry point. Link a query to relevant schema elements.

        Args:
            query: natural language question
            schema: Schema object for the target database
            top_k: Optional retrieval budget (e.g., from QCE)

        Returns:
            SchemaLinkingResult with relevant tables and columns
        """
        # Step 1: Encode query
        query_vec = self._encode_query(query)

        # Step 2: Get schema embeddings (cached)
        schema_data = self._encode_schema(schema)

        # Step 3: Score tables
        table_scores = self._score_tables(
            query_vec, schema, schema_data
        )

        # Step 4: Score columns
        column_scores = self._score_columns(
            query_vec, schema, schema_data, table_scores
        )

        sorted_tables = sorted(table_scores, key=lambda x: x.score, reverse=True)

        # Step 5: Filter by top_k budget or threshold
        if top_k is not None and top_k > 0:
            relevant_tables = sorted_tables[:min(top_k, len(sorted_tables))]
        else:
            relevant_tables = [ts for ts in sorted_tables if ts.score >= self.threshold]

        relevant_columns = [cs for cs in column_scores if cs.score >= self.threshold]

        # Step 6: Fallback -- if nothing passes threshold, take top-2 tables
        if not relevant_tables:
            logger.warning(
                f"No tables above threshold {self.threshold} for query: '{query[:50]}'. "
                f"Using top-2 fallback."
            )
            relevant_tables = sorted_tables[:min(2, len(sorted_tables))]

        # Sort by score descending
        relevant_tables  = sorted(relevant_tables,  key=lambda x: x.score, reverse=True)
        relevant_columns = sorted(relevant_columns, key=lambda x: x.score, reverse=True)

        return SchemaLinkingResult(
            query=query,
            relevant_tables=relevant_tables,
            relevant_columns=relevant_columns,
            all_table_scores=sorted(table_scores,  key=lambda x: x.score, reverse=True),
            all_column_scores=sorted(column_scores, key=lambda x: x.score, reverse=True),
        )

    def _score_tables(
        self,
        query_vec: np.ndarray,
        schema: Schema,
        schema_data: Dict
    ) -> List[TableScore]:
        """Compute relevance score for each table."""
        if len(schema_data["table_embeddings"]) == 0:
            return []

        sims = self._cosine_sim(query_vec, schema_data["table_embeddings"])

        table_scores = []
        for idx, (table, sim) in enumerate(zip(schema.tables, sims)):
            # For tables, score = sim directly (alpha handled at column level)
            score = float(sim)
            table_scores.append(TableScore(
                table=table,
                score=score,
                sim_to_query=float(sim)
            ))
        return table_scores

    def _score_columns(
        self,
        query_vec: np.ndarray,
        schema: Schema,
        schema_data: Dict,
        table_scores: List[TableScore]
    ) -> List[ColumnScore]:
        """
        Compute relevance score for each column.
        Score = alpha * table_sim + beta * column_sim
        This rewards columns that belong to highly relevant tables.
        """
        if len(schema_data["column_embeddings"]) == 0:
            return []

        # Build table_id -> table_score lookup
        table_score_map = {ts.table.table_id: ts.score for ts in table_scores}

        col_sims = self._cosine_sim(query_vec, schema_data["column_embeddings"])

        column_scores = []
        for col, table, col_sim in zip(
            schema_data["valid_columns"],
            schema_data["col_to_table"],
            col_sims
        ):
            table_sim = table_score_map.get(table.table_id, 0.0)

            # Combined score: weighted sum of table sim and column sim
            combined_score = self.alpha * table_sim + self.beta * float(col_sim)

            column_scores.append(ColumnScore(
                column=col,
                table=table,
                score=combined_score,
                sim_to_query=float(col_sim)
            ))

        return column_scores

    # -- Batch Processing -------------------------------------

    def link_batch(
        self,
        samples: List[SpiderSample],
        schemas: Dict[str, Schema],
        verbose: bool = False
    ) -> List[SchemaLinkingResult]:
        """
        Process multiple samples at once.
        Useful for evaluation runs.
        """
        results = []
        for i, sample in enumerate(samples):
            schema = schemas.get(sample.db_id)
            if schema is None:
                logger.warning(f"Schema not found for db_id: {sample.db_id}")
                continue
            result = self.link(sample.question, schema)
            results.append(result)
            if verbose and i % 100 == 0:
                logger.info(f"  Linked {i+1}/{len(samples)} samples...")
        return results

    # -- Evaluation Helper ------------------------------------

    def evaluate_linking_accuracy(
        self,
        samples: List[SpiderSample],
        schemas: Dict[str, Schema],
        top_k: int = 100
    ) -> Dict:
        """
        Evaluate how well schema linking recalls the gold tables.
        A table is 'gold' if it appears in the ground-truth SQL.

        Returns recall@k -- what % of gold tables were correctly retrieved.
        """
        from src.data_loader import SpiderSample
        import re

        total_gold = 0
        total_recalled = 0
        total_samples = 0

        for sample in samples[:top_k]:
            schema = schemas.get(sample.db_id)
            if not schema:
                continue

            # Extract gold table names from SQL (simple heuristic)
            sql_upper = sample.query.upper()
            gold_tables = set()
            for table in schema.tables:
                if table.name_original.upper() in sql_upper:
                    gold_tables.add(table.table_id)

            if not gold_tables:
                continue

            result = self.link(sample.question, schema)
            retrieved_ids = set(result.get_table_ids())

            recalled = len(gold_tables & retrieved_ids)
            total_gold     += len(gold_tables)
            total_recalled += recalled
            total_samples  += 1

        recall = total_recalled / total_gold if total_gold > 0 else 0.0
        return {
            "samples_evaluated": total_samples,
            "table_recall":      round(recall * 100, 2),
            "total_gold_tables": total_gold,
            "total_recalled":    total_recalled,
        }

    def print_result(self, result: SchemaLinkingResult, show_all: bool = False):
        """Pretty print a schema linking result."""
        print("\n" + "="*65)
        print(f"Query: {result.query}")
        print("-"*65)
        print(f"[OK] Relevant Tables ({len(result.relevant_tables)}):")
        for ts in result.relevant_tables:
            print(f"   [{ts.score:.3f}] {ts.table.name_original}")

        print(f"\n[OK] Relevant Columns ({len(result.relevant_columns)}):")
        for cs in result.relevant_columns[:10]:  # top 10
            print(f"   [{cs.score:.3f}] {cs.table.name_original}.{cs.column.name_original}")

        if show_all:
            print(f"\n  All Table Scores:")
            for ts in result.all_table_scores:
                marker = "[OK]" if ts.score >= self.threshold else "[FAIL]"
                print(f"   {marker} [{ts.score:.3f}] {ts.table.name_original}")
        print("="*65)