import os
import json
import sqlite3

bird_dir = os.path.join(os.path.dirname(__file__), "data", "bird", "mini_dev", "MINIDEV")
db_dir = os.path.join(bird_dir, "dev_databases", "financial")
os.makedirs(db_dir, exist_ok=True)

# 1. SQLite DB
conn = sqlite3.connect(os.path.join(db_dir, "financial.sqlite"))
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS client (
    client_id INTEGER PRIMARY KEY,
    gender TEXT,
    birth_date TEXT,
    district_id INTEGER
);
""")
cur.execute("""
CREATE TABLE IF NOT EXISTS account (
    account_id INTEGER PRIMARY KEY,
    district_id INTEGER,
    frequency TEXT,
    date TEXT
);
""")
conn.commit()
conn.close()

# 2. dev_tables.json
tables_data = [
    {
        "db_id": "financial",
        "table_names": ["client", "account"],
        "table_names_original": ["client", "account"],
        "column_names": [
            [-1, "*"],
            [0, "client id"], [0, "gender"], [0, "birth date"], [0, "district id"],
            [1, "account id"], [1, "district id"], [1, "frequency"], [1, "date"]
        ],
        "column_names_original": [
            [-1, "*"],
            [0, "client_id"], [0, "gender"], [0, "birth_date"], [0, "district_id"],
            [1, "account_id"], [1, "district_id"], [1, "frequency"], [1, "date"]
        ],
        "column_types": [
            "others",
            "number", "text", "text", "number",
            "number", "number", "text", "text"
        ],
        "primary_keys": [1, 5],
        "foreign_keys": []
    }
]
with open(os.path.join(bird_dir, "dev_tables.json"), "w", encoding="utf-8") as f:
    json.dump(tables_data, f, indent=2)

# 3. mini_dev_sqlite.json
samples = [
    {
        "question_id": 0,
        "db_id": "financial",
        "question": "How many male clients are there?",
        "evidence": "gender = M refers to male clients",
        "SQL": "SELECT count(*) FROM client WHERE gender = 'M';",
        "difficulty": "simple"
    }
]
with open(os.path.join(bird_dir, "mini_dev_sqlite.json"), "w", encoding="utf-8") as f:
    json.dump(samples, f, indent=2)

print("Sample BIRD Mini-Dev dataset created successfully!")
