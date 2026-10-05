import argparse
import json
import os
import sqlite3
from typing import Any, List, Tuple
import sqlglot
from sqlglot import parse_one


def normalize_sql(sql: str) -> str:

    if not sql:
        return ""

    try:
        expr = parse_one(sql, read="sqlite")
        return expr.sql(dialect="sqlite")

    except Exception:
        return " ".join(sql.strip().lower().split())


def execute_sql(db_path: str, sql: str) -> Tuple[bool, List[Tuple[Any, ...]]]:
    if sql is None:
        return False, []
    sql = sql.strip()

    if not sql.lower().startswith("select"):
        return False, []
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        # convert rows to comparable form
        norm_rows = [tuple(None if v is None else v for v in r) for r in rows]
        # sort for order-insensitive comparison
        try:
            norm_rows_sorted = sorted(norm_rows)
        except Exception:
            norm_rows_sorted = norm_rows
        return True, norm_rows_sorted
    except Exception:
        return False, []
    
    
def get_db_path(data_root: str, db_id: str) -> str:
    """
    Returns SQLite database path for Spider or BIRD.
    """

    spider_db = os.path.join(
        data_root,
        "database",
        db_id,
        f"{db_id}.sqlite",
    )

    if os.path.exists(spider_db):
        return spider_db

    bird_db = os.path.join(
        data_root,
        "dev_databases",
        db_id,
        f"{db_id}.sqlite",
    )

    return bird_db

def extract_tables(sql: str) -> List[str]:
    try:
        expr = parse_one(sql, read="sqlite")

        tables = set()

        for table in expr.find_all(sqlglot.expressions.Table):
            if table.name:
                tables.add(table.name.lower())

        return sorted(tables)

    except Exception:
        return []


def evaluate(predictions_path: str, data_root: str, out_path: str = None) -> dict:
    with open(predictions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
         preds = data
    else:
        preds = data["predictions"] if isinstance(data, dict) else data
    total = len(preds)

    em_correct = 0
    ex_correct = 0
    ex_attempted = 0
    join_path_correct = 0
    valid_queries = 0
    # prepare per-complexity accumulators
    complexity_buckets = {
        "aggregation queries": [],
        "single-table": [],
        "nested queries": [],
        "2-table joins": [],
        "3-table joins": [],
        "4+ table joins": [],
    }

    def classify(sql: str):
        # heuristics based on gold SQL
        s = (sql or "").lower()
        tables = extract_tables(sql or "")
        num_tables = len(tables)
        is_agg = any(fn in s for fn in ["count(", "avg(", "min(", "max(", "sum("])
        is_nested = "select" in s and "select" in s[s.find("select") + 6:]

        if is_agg:
            return "aggregation queries"
        if is_nested:
            return "nested queries"
        if num_tables <= 1:
            return "single-table"
        if num_tables == 2:
            return "2-table joins"
        if num_tables == 3:
            return "3-table joins"
        return "4+ table joins"

    for item in preds:
        db_id = item.get("db_id")
        gold_sql = item.get("gold_sql")
        pred_sql = item.get("predicted_sql")
        pred_join = item.get("join_path", []) or []
        if item.get("valid_sql"):
            valid_queries += 1

        # Exact Match (normalize both)
        n_gold = normalize_sql(gold_sql or "")
        n_pred = normalize_sql(pred_sql or "")
        if n_gold == n_pred:
            em_correct += 1
        
        # Execution Accuracy -- attempt execution on sqlite DB
        db_file = get_db_path(data_root, db_id)
        if os.path.exists(db_file):
            ok_pred, pred_rows = execute_sql(db_file, pred_sql)
            ok_gold, gold_rows = execute_sql(db_file, gold_sql)
            if ok_pred:
                ex_attempted += 1
                if ok_gold and pred_rows == gold_rows:
                    ex_correct += 1

        # Join Path accuracy -- compare predicted join path tables to tables in gold SQL
        gold_tables = extract_tables(gold_sql or "")
        pred_tables = []

        for p in pred_join:

            if isinstance(p, str):
                pred_tables.append(p.lower())

            elif isinstance(p, dict):

                if "from" in p:
                    pred_tables.append(p["from"].lower())

                if "to" in p:
                    pred_tables.append(p["to"].lower())
        if set(pred_tables) == set(gold_tables) and len(gold_tables) > 0:
            join_path_correct += 1

        # classify and append item for per-complexity metrics
        bucket = classify(gold_sql)
        complexity_buckets.setdefault(bucket, []).append(item)

    em_pct = 100.0 * em_correct / total if total else 0.0
    ex_pct_overall = 100.0 * ex_correct / total if total else 0.0
    ex_pct_attempted = 100.0 * ex_correct / ex_attempted if ex_attempted else 0.0
    join_path_pct = 100.0 * join_path_correct / total if total else 0.0

    results = {
        "total": total,
        "exact_match": em_correct,
        "exact_match_pct": round(em_pct, 2),
        "execution_correct": ex_correct,
        "execution_pct_overall": round(ex_pct_overall, 2),
        "execution_pct_attempted": round(ex_pct_attempted, 2),
        "ex_attempted": ex_attempted,
        "join_path_correct": join_path_correct,
        "join_path_pct": round(join_path_pct, 2),
        "valid_query_rate": round(
            100 * valid_queries / total,
            2,
        ),
    }

    # per-complexity breakdown
    per_complexity = {}
    for name, items in complexity_buckets.items():
        t = len(items)
        em_c = 0
        ex_c = 0
        vqr_c = 0
        ex_attempted_c = 0
        for it in items:
            g = it.get("gold_sql")
            p = it.get("predicted_sql")
            valid = bool(it.get("valid_sql"))
            if normalize_sql(g or "") == normalize_sql(p or ""):
                em_c += 1
            # execution
            db_file = get_db_path(
                data_root,
                it.get("db_id"),
        )
            if os.path.exists(db_file):
                ok_pred, pred_rows = execute_sql(db_file, p)
                ok_gold, gold_rows = execute_sql(db_file, g)
                if ok_pred:
                    ex_attempted_c += 1
                    if ok_gold and pred_rows == gold_rows:
                        ex_c += 1
            if valid:
                vqr_c += 1

        per_complexity[name] = {
            "total": t,
            "EM (%)": round(100.0 * em_c / t, 2) if t else 0.0,
            "EX (%)": round(100.0 * ex_c / t, 2) if t else 0.0,
            "VQR (%)": round(100.0 * vqr_c / t, 2) if t else 0.0,
        }

    results["per_complexity"] = per_complexity

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset": os.path.basename(data_root),
                    "metrics": results,
                },
                f,
                indent=2,
            )

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate EX/EM/Join Path accuracy from predictions JSON")
    parser.add_argument("predictions", help="Path to predictions JSON (outputs/results/*.json)")
    parser.add_argument("data_root", help="Path to dataset root (e.g., data/spider)")
    parser.add_argument("--out", help="Optional output JSON file to write metrics")

    args = parser.parse_args()
    res = evaluate(args.predictions, args.data_root, args.out)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
