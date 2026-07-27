# ConSQL: Graph-Guided Context Optimization for Text-to-SQL using Local LLMs

ConSQL is a Text-to-SQL framework that converts natural language questions into executable SQL queries using **graph-guided schema optimization** and **local Large Language Models (LLMs)** powered by **Ollama**.

Instead of providing the entire database schema to the language model, ConSQL constructs a relational graph from primary key and foreign key relationships, identifies only the relevant schema components for a given query, and generates optimized SQL statements. This approach reduces prompt size, improves relational reasoning, and enables fully local execution without requiring cloud-based APIs.

---

## Features

- Graph-guided schema representation
- Query-aware schema pruning
- Dynamic prompt generation
- Local inference using Ollama
- No API keys required
- SQLite database support
- Spider benchmark compatible
- Modular and extensible architecture

---

# Project Workflow

```
Natural Language Question
            │
            ▼
Load SQLite Database
            │
            ▼
Extract Schema Information
            │
            ▼
Build Relational Graph
            │
            ▼
Query-aware Schema Selection
            │
            ▼
Prompt Construction
            │
            ▼
Ollama Local LLM
            │
            ▼
Generate SQL Query
            │
            ▼
SQL Validation
            │
            ▼
Execute SQL
            │
            ▼
Display Results
```

---

# Project Structure

```
ConSQL/
│
├── data/                 # Spider dataset and SQLite databases
├── src/                  # Source code
│   ├── llm/              # Ollama client
│   ├── graph/            # Graph construction and traversal
│   ├── prompts/          # Prompt generation
│   ├── sql/              # SQL generation and validation
│   ├── execution/        # SQL execution
│   └── evaluation/       # Evaluation scripts
│
├── main.py               # Entry point
├── requirements.txt
└── README.md
```

> **Note:** Update the project structure above to match your repository.

---

# Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.10 or later
- Ollama
- Git

---

# Installation

## Clone the repository

```bash
git clone https://github.com/<your-username>/ConSQL.git

cd ConSQL
```

---

## Create a virtual environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install dependencies

```bash
pip install -r requirements.txt
```

---

# Install Ollama

Download and install Ollama from:

https://ollama.com/download

Verify the installation:

```bash
ollama --version
```

Download your preferred model:

```bash
ollama pull llama3
```

or

```bash
ollama pull qwen2.5-coder
```

Start the Ollama server:

```bash
ollama serve
```

---

# Dataset Setup

Download the Spider dataset and organize it as follows:

```
data/
└── spider/
    ├── database/
    ├── tables.json
    ├── train_spider.json
    └── dev.json
```

---

# Running the Project

After starting the Ollama server, run:

```bash
python main.py
```

Example input:

```
Which department has the highest average salary?
```

Example output:

```sql
SELECT d.department_name,
AVG(s.salary)
FROM department d
JOIN employee e
ON d.department_id = e.department_id
JOIN salary s
ON e.employee_id = s.employee_id
GROUP BY d.department_name
ORDER BY AVG(s.salary) DESC
LIMIT 1;
```

---

# How ConSQL Works

### Step 1 – User Query

The user submits a question in natural language.

### Step 2 – Schema Extraction

The system extracts tables, columns, primary keys, and foreign keys from the SQLite database.

### Step 3 – Graph Construction

A relational graph is built where:

- Nodes represent database tables.
- Edges represent relationships between tables.

### Step 4 – Context Optimization

The graph is traversed to identify only the schema elements relevant to the user query.

### Step 5 – Prompt Generation

A prompt is created using:

- User question
- Relevant tables
- Columns
- Relationships
- SQL generation instructions

### Step 6 – SQL Generation

The optimized prompt is sent to the local LLM through Ollama.

### Step 7 – SQL Validation

The generated SQL is validated and cleaned before execution.

### Step 8 – Query Execution

The validated SQL is executed on the SQLite database, and the results are displayed.

---

# Technologies Used

| Component | Technology |
|-----------|------------|
| Programming Language | Python |
| Database | SQLite |
| Local LLM | Ollama |
| Graph Processing | NetworkX |
| Dataset | Spider |
| SQL Execution | sqlite3 |

---

# Configuration

Configure the model and project settings in your configuration file (for example, `config.py` or `.env`).

Example:

```python
LLM_PROVIDER = "ollama"
LLM_MODEL = "llama3"
```

---

# License

This project is intended for academic and research purposes.

---

# Acknowledgements

This project makes use of the following open-source resources:

- Spider Benchmark Dataset
- Ollama
- SQLite
- NetworkX