"""
Global configuration for ConSQL.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------
# Project Paths
# -----------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "results"

# -----------------------------------------------------
# Spider
# -----------------------------------------------------

SPIDER_DIR = DATA_DIR / "spider"
SPIDER_DB_DIR = SPIDER_DIR / "database"

# -----------------------------------------------------
# BIRD
# -----------------------------------------------------

BIRD_DIR = DATA_DIR / "bird"

BIRD_MINIDEV_DIR = BIRD_DIR / "mini_dev" / "MINIDEV"

BIRD_DB_DIR = BIRD_MINIDEV_DIR / "dev_databases"

# -----------------------------------------------------
# Ollama
# -----------------------------------------------------

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "deepseek-coder:6.7b"
)

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)

# -----------------------------------------------------
# Generation
# -----------------------------------------------------

TEMPERATURE = float(
    os.getenv("TEMPERATURE", "0.0")
)

TOP_P = float(
    os.getenv("TOP_P", "0.95")
)

MAX_OUTPUT_TOKENS = int(
    os.getenv("MAX_OUTPUT_TOKENS", "1024")
)

# -----------------------------------------------------
# Available Models
# -----------------------------------------------------

AVAILABLE_MODELS = {
    "ollama": [
        "deepseek-coder:6.7b",
        "qwen2.5-coder:7b",
        "codellama:7b",
    ]
}