# src/bird_loader.py
# ============================================================
# ConSQL - BIRD Mini-Dev Loader
# Exact paths confirmed from dir output:
#   data/bird/minidev/MINIDEV/mini_dev_sqlite.json   <- samples
#   data/bird/minidev/MINIDEV/dev_tables.json        <- schemas
#   data/bird/minidev/MINIDEV/dev_databases/<db>/<db>.sqlite
# ============================================================

import json
import os
import logging
from typing import Dict, List
from dataclasses import dataclass

from src.config import BIRD_MINIDEV_DIR
from src.data_loader import ( Schema, Table, Column, ForeignKey)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
print("="*60)
print("BIRD_MINIDEV_DIR =", BIRD_MINIDEV_DIR)
print("="*60)

@dataclass
class BIRDSample:
    """One BIRD Mini-Dev example."""
    db_id:       str
    question:    str
    query:       str       # ground truth SQL
    evidence:    str       # external hint (BIRD-specific)
    difficulty:  str       # simple / moderate / challenging
    question_id: int = 0

    def to_query_with_evidence(self) -> str:
        """Combine question + evidence for richer SBERT encoding."""
        if self.evidence and self.evidence.strip():
            return f"{self.question} {self.evidence}"
        return self.question


class BIRDMiniDevLoader:
    """
    Loads BIRD Mini-Dev dataset.

    Confirmed structure:
      data/bird/minidev/MINIDEV/
        mini_dev_sqlite.json     <- 500 samples
        dev_tables.json          <- 11 database schemas
        dev_databases/
          california_schools/california_schools.sqlite
          card_games/card_games.sqlite
          ... (11 databases total)
    """

    def __init__(self, bird_minidev_dir: str):
        self.bird_minidev_dir = bird_minidev_dir
        self.schemas: Dict[str, Schema] = {}
        self.dev_samples: List[BIRDSample] = []
        self._validate()

    def _validate(self):
        required = [
            os.path.join(self.bird_minidev_dir, "mini_dev_sqlite.json"),
            os.path.join(self.bird_minidev_dir, "dev_tables.json"),
            os.path.join(self.bird_minidev_dir, "dev_databases"),
        ]
        missing = [p for p in required if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"\n[ERROR] BIRD Mini-Dev files not found:\n" +
                "\n".join(f"  {p}" for p in missing)
            )
        logger.info("✅ All BIRD Mini-Dev files found.")

    def load_schemas(self) -> Dict[str, Schema]:
        """Load all 11 BIRD Mini-Dev database schemas."""
        tables_path = os.path.join(self.bird_minidev_dir, "dev_tables.json")
        logger.info(f"Loading BIRD schemas from {tables_path} ...")

        with open(tables_path, encoding="utf-8") as f:
            raw_schemas = json.load(f)

        for raw in raw_schemas:
            schema = self._parse_schema(raw)
            self.schemas[schema.db_id] = schema

        logger.info(f"✅ Loaded {len(self.schemas)} BIRD Mini-Dev schemas.")
        return self.schemas

    def _parse_schema(self, raw: dict) -> Schema:
        """Parse BIRD schema — same format as Spider."""
        db_id = raw["db_id"]

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

        all_columns = []
        for col_id, ((table_id, col_name), (_, col_name_orig), col_type) in enumerate(
            zip(raw["column_names"], raw["column_names_original"], raw["column_types"])
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
            if 0 <= table_id < len(tables):
                tables[table_id].columns.append(col)

        for pk_entry in raw.get("primary_keys", []):
            if isinstance(pk_entry, list):
                for pk_id in pk_entry:
                    if isinstance(pk_id, int) and pk_id < len(all_columns):
                        col = all_columns[pk_id]
                        if 0 <= col.table_id < len(tables):
                            tables[col.table_id].primary_keys.append(pk_id)
            elif isinstance(pk_entry, int):
                pk_id = pk_entry
                if pk_id < len(all_columns):
                    col = all_columns[pk_id]
                    if 0 <= col.table_id < len(tables):
                        tables[col.table_id].primary_keys.append(pk_id)

        foreign_keys = []
        for from_col_id, to_col_id in raw.get("foreign_keys", []):
            if from_col_id < len(all_columns) and to_col_id < len(all_columns):
                fc = all_columns[from_col_id]
                tc = all_columns[to_col_id]
                foreign_keys.append(ForeignKey(
                    from_col_id=from_col_id, to_col_id=to_col_id,
                    from_table_id=fc.table_id, to_table_id=tc.table_id
                ))

        return Schema(db_id=db_id, tables=tables,
                      foreign_keys=foreign_keys, columns=all_columns)
    
    def load_dev(self) -> List[BIRDSample]:
        """Load 500 Mini-Dev samples from mini_dev_sqlite.json."""
        path = os.path.join(self.bird_minidev_dir, "mini_dev_sqlite.json")
        logger.info(f"Loading BIRD Mini-Dev samples from {path} ...")

        with open(path, encoding="utf-8") as f:
            raw_data = json.load(f)

        # mini_dev_sqlite.json can be a list or dict with a key
        if isinstance(raw_data, list):
            raw_samples = raw_data

        elif "data" in raw_data:
            raw_samples = raw_data["data"]

        elif "questions" in raw_data:
            raw_samples = raw_data["questions"]

        else:
            raw_samples = list(raw_data.values())[0]

        samples = []
        for i, raw in enumerate(raw_samples):
            # Handle both field name variants seen in BIRD
            sql = raw.get("SQL", raw.get("query", raw.get("sql", "")))
            sample = BIRDSample(
                db_id=raw.get("db_id", ""),
                question=raw.get("question", "").strip(),
                query=sql.strip(),
                evidence=raw.get("evidence", "").strip(),
                difficulty=raw.get("difficulty", "simple"),
                question_id=raw.get("question_id", i)
            )
            samples.append(sample)

        self.dev_samples = samples
        logger.info(f"✅ Loaded {len(samples)} BIRD Mini-Dev samples.")
        return self.dev_samples
    def load(self):
        """
        Load schemas and Mini-Dev samples.
        """

        self.load_schemas()
        self.load_dev()

        return self.schemas, self.dev_samples
    def get_stats(self) -> dict:
        difficulties = {}
        for s in self.dev_samples:
            difficulties[s.difficulty] = difficulties.get(s.difficulty, 0) + 1

        with_evidence = sum(1 for s in self.dev_samples if s.evidence.strip())

        return {
            "num_schemas":       len(self.schemas),
            "num_dev_samples":   len(self.dev_samples),
            "num_databases": len(self.schemas),
            "num_questions": len(self.dev_samples),
            "avg_tables_per_db": round(
                sum(len(s.tables) for s in self.schemas.values()) /
                max(len(self.schemas), 1), 2
            ),
            "avg_fks_per_db": round(
                sum(len(s.foreign_keys) for s in self.schemas.values()) /
                max(len(self.schemas), 1), 2
            ),
            "difficulty_breakdown": difficulties,
            "samples_with_evidence": with_evidence,
            "databases": list(self.schemas.keys()),
        }
    def get_database_path(self, db_id: str) -> str:
        """
        Returns the SQLite database path for a BIRD database.
        """

        return os.path.join(
            self.bird_minidev_dir,
            "dev_databases",
            db_id,
            f"{db_id}.sqlite",
    )
        

    def print_sample(self, sample: BIRDSample):
        print(f"\n{'='*60}")
        print(f"DB         : {sample.db_id}")
        print(f"Question   : {sample.question}")
        print(f"Evidence   : {sample.evidence[:80] if sample.evidence else 'None'}")
        print(f"SQL        : {sample.query}")
        print(f"Difficulty : {sample.difficulty}")
        print(f"{'='*60}")