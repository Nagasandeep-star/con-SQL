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
# LLM Providers Configuration
# -----------------------------------------------------

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-pro")

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# -----------------------------------------------------
# Generation
# -----------------------------------------------------

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
TOP_P = float(os.getenv("TOP_P", "0.95"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1024"))

# -----------------------------------------------------
# QCE & Relational Optimization Settings
# -----------------------------------------------------

QCE_MIN_BUDGET = int(os.getenv("QCE_MIN_BUDGET", "3"))
QCE_MAX_BUDGET = int(os.getenv("QCE_MAX_BUDGET", "8"))
JOIN_NECESSITY_THRESHOLD = float(os.getenv("JOIN_NECESSITY_THRESHOLD", "0.35"))
MAX_JOIN_DEPTH = int(os.getenv("MAX_JOIN_DEPTH", "4"))
NUM_SERIALIZED_PATHS = int(os.getenv("NUM_SERIALIZED_PATHS", "1"))  # 1 for Top-1, 2-3 for Multi-Path/Beam

# Available Models across providers
AVAILABLE_MODELS = {
    "gemini": [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-1.5-pro",
    ],
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-3.5-turbo",
    ],
    "ollama": [
        "qwen2.5-coder:7b",
        "qwen2.5-coder:32b",
        "llama3.1:8b",
        "deepseek-coder:6.7b",
        "codellama:7b",
    ],
    "mock": [
        "mock-fast",
    ]
}