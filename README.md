# ConSQL: Equivalence-Aware Relational Context Optimization and Multi-Candidate Graph Guidance for Text-to-SQL Generation

ConSQL is a high-performance Text-to-SQL framework that inserts an explicit, learning-based relational optimization stage between schema retrieval and Large Language Model (LLM) generation.

Instead of presenting the entire database catalog or relying on unguided LLM reasoning over raw schema elements, ConSQL:
1. **Estimates Query Complexity (QCE)** to adaptively allocate the schema retrieval budget ($K \in [3, 8]$ tables), preventing context starvation on multi-table queries and prompt bloat on simple queries.
2. **Constructs a Relational Schema Graph** and scores candidate join paths using **Equivalence-Aware Relational Context Optimization (E-LRCO)** via a calibrated LambdaMART ranker.
3. **Serializes Multi-Candidate Beam Guidance**, providing the LLM with structurally coherent, ranked relational paths to eliminate brittle single-path bottlenecks.
4. **Supports Multiple LLM Backbones** (Google Gemini 2.5 Pro, OpenAI GPT-4o, Ollama local open-source models like Qwen-2.5-Coder, and deterministic Mock LLM for offline testing).
5. **Validates Generated Queries** with two-stage syntactic and schema-consistency checking with automated correction feedback loops.

---

## Key Features

- **Multi-Backbone Support**: Seamless integration with Google Gemini (`gemini-2.5-pro`), OpenAI (`gpt-4o`), Ollama (`qwen2.5-coder:32b`, `llama3`), or offline Mock execution.
- **Equivalence-Aware Join Ranking (E-LRCO)**: Graded relevance supervision against the relational equivalence closure $\mathcal{J}_q^*$, eliminating label noise from commutative joins and isomorphic foreign keys.
- **Multi-Candidate Beam Serialization**: Supports Top-$K$ candidate join paths ($K \in \{1, 2, 3\}$) with normalized confidence weights, overcoming single-path bottlenecks.
- **Oracle Join-Path Upper Bounding**: Built-in `--oracle` extractor that isolates relational context optimization errors from downstream LLM reasoning bottlenecks.
- **Direct QCE Evaluation**: Evaluate query complexity correlation ($r=0.831$) and compare adaptive retrieval directly against static budgets ($K \in \{3, 5, 8, 12\}$).
- **Statistical Significance Suite**: Automated 5-run evaluation reporting mean $\pm$ std, 95% bootstrap confidence intervals, and paired McNemar's tests with Bonferroni correction.
- **Granular Latency Profiling**: Explicit decoupling of Pre-LLM Relational Context Optimization latency ($T_{\mathrm{RCO}} \approx 112$\,ms) from LLM token decoding latency ($T_{\mathrm{LLM}} \approx 1.5$\,s).

---

## Architecture Overview

```
Natural Language Query (q) + Schema (S)
                  │
                  ▼
   [Query Complexity Estimation (QCE)]
        ├── Structural Feature Extraction
        └── Adaptive Budget K in [3, 8]
                  │
                  ▼
   [Semantic Schema Retrieval (SBERT)]
        └── Adaptive Top-K Subgraph R_S
                  │
                  ▼
   [Relational Graph & Path Enumeration]
        └── Bounded DFS Candidate Paths P
                  │
                  ▼
   [Equivalence-Aware LRCO (LambdaMART)]
        └── Candidate Scores & Multi-Candidate Beam Selection
                  │
                  ▼
   [Relational Context Serialization]
        └── Structured Prompt with Ranked Path Guidance
                  │
                  ▼
   [LLM Generation (Gemini / GPT-4o / Qwen / Mock)]
                  │
                  ▼
   [Two-Stage Validation (Syntax + Schema)]
        └── Execution / Corrective Regeneration Feedback Loop
```

---

## Installation & Setup

### 1. Requirements
- Python 3.10 to 3.14
- SQLite3
- Optional: Google GenAI API key (for Gemini), OpenAI API key (for GPT-4o), or Ollama (for local LLMs).

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Setup
Run the diagnostic setup script:
```bash
python check_setup.py
```

---

## Running Evaluations

### 1. Standard ConSQL Evaluation
```bash
python -m src.run_evaluation --provider gemini --model gemini-2.5-pro --max_samples 100
```

### 2. Cross-Backbone Evaluation
Evaluate with OpenAI GPT-4o:
```bash
python -m src.run_evaluation --provider openai --model gpt-4o --max_samples 100
```
Evaluate with a local Ollama model (e.g., Qwen-2.5-Coder):
```bash
python -m src.run_evaluation --provider ollama --model qwen2.5-coder:32b --max_samples 100
```
Fast offline testing using deterministic Mock LLM:
```bash
python -m src.run_evaluation --provider mock --max_samples 50
```

### 3. Multi-Candidate Beam Serialization
Serialize Top-2 candidate paths to alleviate single-path bottlenecks:
```bash
python -m src.run_evaluation --provider gemini --multi_path --max_samples 100
```

### 4. Oracle Join-Path Upper Bound
Run ConSQL with oracle ground-truth join paths to isolate relational ranking error from downstream LLM reasoning:
```bash
python -m src.run_evaluation --provider gemini --oracle --max_samples 100
```

### 5. Direct QCE Evaluation
Directly evaluate complexity estimation accuracy and compare adaptive retrieval against fixed budgets ($K \in \{3, 5, 8, 12\}$):
```bash
python -m src.qce.evaluate_qce --dataset_path data/spider/dev.json
```

### 6. Statistical Significance & 5-Run Variance
Compute 5-run empirical variance ($\mu \pm \sigma$), 95% bootstrap confidence intervals, and paired McNemar tests with Bonferroni correction:
```bash
python -m src.evaluator.statistical_test
```

---

## Test Suite

Run all verified unit and integration tests:
```bash
python -m unittest discover -s tests
```
All 23 test suites (18 core pipeline tests + 5 enhancement tests) pass cleanly.

---

## Citation

```bibtex
@article{reddy2026consql,
  title={ConSQL: Equivalence-Aware Relational Context Optimization and Multi-Candidate Graph Guidance for Text-to-SQL Generation},
  author={Reddy, Veerababu and Uppukonda, Usha Rani and Tabassum, Shaik and Sri, Ghanta Pooja and Sandeep, Thiruveedula Naga and Naik, Megavath Dattu},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2026}
}
```