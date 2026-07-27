# src/data_loader.py
# ============================================================
# ConSQL - Data Loader
# Loads and preprocesses Spider (and later BIRD) datasets
# ============================================================

import json
import os
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────

@dataclass
class Column:
    """Represents a single column in a database table."""
    col_id: int
    name: str               # original name e.g. "student_id"
    name_original: str      # exact original name from schema
    table_id: int
    col_type: str           # text, number, time, boolean, others
    is_primary_key: bool = False

    def __str__(self):
        return f"{self.name} ({self.col_type})"


@dataclass
class ForeignKey:
    """Represents a FK relationship between two columns."""
    from_col_id: int
    to_col_id: int
    from_table_id: int
    to_table_id: int


@dataclass
class Table:
    """Represents a database table with its columns."""
    table_id: int
    name: str               # normalized name e.g. "student"
    name_original: str      # exact original name
    columns: List[Column] = field(default_factory=list)
    primary_keys: List[int] = field(default_factory=list)

    def __str__(self):
        return f"Table({self.name}, cols={[c.name for c in self.columns]})"


@dataclass
class Schema:
    """Represents a complete database schema."""
    db_id: str
    tables: List[Table] = field(default_factory=list)
    foreign_keys: List[ForeignKey] = field(default_factory=list)
    columns: List[Column] = field(default_factory=list)

    def get_table_by_id(self, table_id: int) -> Optional[Table]:
        for t in self.tables:
            if t.table_id == table_id:
                return t
        return None

    def get_table_by_name(self, name: str) -> Optional[Table]:
        for t in self.tables:
            if t.name.lower() == name.lower():
                return t
        return None

    def get_column_by_id(self, col_id: int) -> Optional[Column]:
        for c in self.columns:
            if c.col_id == col_id:
                return c
        return None

    def summary(self) -> str:
        return (f"Schema: {self.db_id} | "
                f"Tables: {len(self.tables)} | "
                f"Columns: {len(self.columns)} | "
                f"FKs: {len(self.foreign_keys)}")


@dataclass
class SpiderSample:
    """Represents a single Spider training/dev example."""
    db_id: str
    question: str           # natural language question
    query: str              # ground truth SQL
    query_toks: List[str]   # tokenized SQL
    question_toks: List[str]# tokenized question
    sql: Dict               # structured SQL parse


# ─────────────────────────────────────────
# SPIDER DATA LOADER
# ─────────────────────────────────────────

class SpiderDataLoader:
    """
    Loads the Spider dataset including:
    - train_spider.json  (training samples)
    - dev.json           (development/validation samples)
    - tables.json        (all database schemas)
    """

    def __init__(self, spider_dir: str):
        self.spider_dir = spider_dir
        self.schemas: Dict[str, Schema] = {}
        self.train_samples: List[SpiderSample] = []
        self.dev_samples: List[SpiderSample] = []

        self._validate_paths()

    def _validate_paths(self):
        """Check all required files exist."""
        required = [
            os.path.join(self.spider_dir, "train_spider.json"),
            os.path.join(self.spider_dir, "dev.json"),
            os.path.join(self.spider_dir, "tables.json"),
        ]
        for path in required:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"\n[ERROR] Required file not found: {path}"
                    f"\nMake sure your Spider dataset is placed in: {self.spider_dir}"
                    f"\nExpected structure:"
                    f"\n  data/spider/"
                    f"\n  ├── train_spider.json"
                    f"\n  ├── dev.json"
                    f"\n  ├── tables.json"
                    f"\n  └── database/"
                )
        logger.info("✅ All Spider files found.")

    # ── Schema Loading ──────────────────────────────────────

    def load_schemas(self) -> Dict[str, Schema]:
        """
        Parse tables.json and build Schema objects.
        Spider tables.json structure:
        [
          {
            "db_id": "...",
            "table_names": [...],
            "table_names_original": [...],
            "column_names": [[table_id, col_name], ...],
            "column_names_original": [[table_id, col_name_original], ...],
            "column_types": [...],
            "primary_keys": [...],
            "foreign_keys": [[from_col_id, to_col_id], ...]
          },
          ...
        ]
        """
        tables_path = os.path.join(self.spider_dir, "tables.json")
        logger.info(f"Loading schemas from {tables_path} ...")

        with open(tables_path, "r", encoding="utf-8") as f:
            raw_schemas = json.load(f)

        for raw in raw_schemas:
            schema = self._parse_schema(raw)
            self.schemas[schema.db_id] = schema

        logger.info(f"✅ Loaded {len(self.schemas)} database schemas.")
        return self.schemas

    def _parse_schema(self, raw: Dict) -> Schema:
        """Convert raw Spider schema dict into Schema dataclass."""
        db_id = raw["db_id"]

        # ── Build Tables ──
        tables = []
        for idx, (name, name_orig) in enumerate(
            zip(raw["table_names"], raw["table_names_original"])
        ):
            tables.append(Table(
                table_id=idx,
                name=name,
                name_original=name_orig,
                columns=[],
                primary_keys=[]
            ))

        # ── Build Columns ──
        # column_names[0] is always ("*", -1) — the wildcard, skip it
        all_columns = []
        for col_id, ((table_id, col_name), (_, col_name_orig), col_type) in enumerate(
            zip(
                raw["column_names"],
                raw["column_names_original"],
                raw["column_types"]
            )
        ):
            col = Column(
                col_id=col_id,
                name=col_name,
                name_original=col_name_orig,
                table_id=table_id,
                col_type=col_type,
                is_primary_key=(col_id in raw.get("primary_keys", []))
            )
            all_columns.append(col)

            # Attach column to its table (skip wildcard col with table_id = -1)
            if table_id >= 0 and table_id < len(tables):
                tables[table_id].columns.append(col)

        # ── Primary Keys ──
        for pk_col_id in raw.get("primary_keys", []):
            col = all_columns[pk_col_id]
            if col.table_id >= 0:
                tables[col.table_id].primary_keys.append(pk_col_id)

        # ── Foreign Keys ──
        foreign_keys = []
        for from_col_id, to_col_id in raw.get("foreign_keys", []):
            from_col = all_columns[from_col_id]
            to_col   = all_columns[to_col_id]
            fk = ForeignKey(
                from_col_id=from_col_id,
                to_col_id=to_col_id,
                from_table_id=from_col.table_id,
                to_table_id=to_col.table_id
            )
            foreign_keys.append(fk)

        return Schema(
            db_id=db_id,
            tables=tables,
            foreign_keys=foreign_keys,
            columns=all_columns
        )

    # ── Sample Loading ───────────────────────────────────────

    def load_train(self) -> List[SpiderSample]:
        path = os.path.join(self.spider_dir, "train_spider.json")
        self.train_samples = self._load_samples(path, split="train")
        return self.train_samples

    def load_dev(self) -> List[SpiderSample]:
        path = os.path.join(self.spider_dir, "dev.json")
        self.dev_samples = self._load_samples(path, split="dev")
        return self.dev_samples

    def _load_samples(self, path: str, split: str) -> List[SpiderSample]:
        logger.info(f"Loading {split} samples from {path} ...")

        with open(path, "r", encoding="utf-8") as f:
            raw_samples = json.load(f)

        samples = []
        skipped = 0
        for raw in raw_samples:
            db_id = raw.get("db_id", "")

            # Skip samples whose schema isn't loaded yet
            if self.schemas and db_id not in self.schemas:
                skipped += 1
                continue

            sample = SpiderSample(
                db_id=db_id,
                question=raw.get("question", "").strip(),
                query=raw.get("query", "").strip(),
                query_toks=raw.get("query_toks", []),
                question_toks=raw.get("question_toks", []),
                sql=raw.get("sql", {})
            )
            samples.append(sample)

        if skipped:
            logger.warning(f"⚠️  Skipped {skipped} samples (schema not found).")
        logger.info(f"✅ Loaded {len(samples)} {split} samples.")
        return samples

    # ── Utilities ────────────────────────────────────────────

    def get_schema(self, db_id: str) -> Optional[Schema]:
        return self.schemas.get(db_id)

    def get_sample_with_schema(self, sample: SpiderSample) -> Tuple[SpiderSample, Schema]:
        schema = self.schemas.get(sample.db_id)
        if schema is None:
            raise ValueError(f"Schema not found for db_id: {sample.db_id}")
        return sample, schema

    def get_stats(self) -> Dict:
        """Return dataset statistics."""
        return {
            "num_schemas":       len(self.schemas),
            "num_train_samples": len(self.train_samples),
            "num_dev_samples":   len(self.dev_samples),
            "avg_tables_per_db": round(
                sum(len(s.tables) for s in self.schemas.values()) / max(len(self.schemas), 1), 2
            ),
            "avg_fks_per_db": round(
                sum(len(s.foreign_keys) for s in self.schemas.values()) / max(len(self.schemas), 1), 2
            ),
            "total_columns": sum(len(s.columns) for s in self.schemas.values()),
        }

    def print_sample(self, sample: SpiderSample, show_schema: bool = True):
        """Pretty print a sample for debugging."""
        print("\n" + "="*60)
        print(f"DB       : {sample.db_id}")
        print(f"Question : {sample.question}")
        print(f"SQL      : {sample.query}")
        if show_schema:
            schema = self.get_schema(sample.db_id)
            if schema:
                print(f"\nSchema   : {schema.summary()}")
                for t in schema.tables:
                    cols = ", ".join(c.name for c in t.columns[:5])
                    print(f"  Table  : {t.name_original} → [{cols}{'...' if len(t.columns)>5 else ''}]")
                if schema.foreign_keys:
                    print(f"\nForeign Keys:")
                    for fk in schema.foreign_keys:
                        fc = schema.get_column_by_id(fk.from_col_id)
                        tc = schema.get_column_by_id(fk.to_col_id)
                        ft = schema.get_table_by_id(fk.from_table_id)
                        tt = schema.get_table_by_id(fk.to_table_id)
                        if fc and tc and ft and tt:
                            print(f"  {ft.name}.{fc.name} → {tt.name}.{tc.name}")
        print("="*60)