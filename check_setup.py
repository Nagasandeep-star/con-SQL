# check_setup.py
# ============================================================
# ConSQL - System Setup and Environment Diagnostic Tool
# Verifies packages, project modules, datasets, and smoke tests
# Usage: python check_setup.py
# ============================================================

import os
import sys

# Ensure both project root and src/ are in sys.path for robust imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
for path in [CURRENT_DIR, SRC_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

print("=" * 60)
print("  ConSQL Setup Checker")
print("=" * 60)

errors = []
warnings = []

# -- 1. Check Python version ----------------------------------
py_version = sys.version.split()[0]
print(f"\n[1] Python version: {py_version}")
if sys.version_info < (3, 10):
    print("   [WARN] Python 3.10+ recommended")
    warnings.append("Consider upgrading Python to 3.10+")
else:
    print("   [OK]   Python version is compatible")

# -- 2. Check all packages ------------------------------------
print("\n[2] Checking packages...")
packages = {
    "torch":                 "PyTorch",
    "transformers":          "HuggingFace Transformers",
    "sentence_transformers": "Sentence Transformers (SBERT)",
    "networkx":              "NetworkX",
    "sqlglot":               "SQLGlot",
    "pandas":                "Pandas",
    "numpy":                 "NumPy",
    "sklearn":               "Scikit-learn",
    "lightgbm":              "LightGBM (LambdaMART)",
    "tqdm":                  "TQDM",
    "gradio":                "Gradio",
    "accelerate":            "Accelerate",
}

for pkg, name in packages.items():
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "?")
        print(f"   [OK]   {name:<35} v{ver}")
    except ImportError:
        print(f"   [FAIL] {name:<35} NOT FOUND")
        errors.append(f"pip install {pkg}")

# Optional check for Google GenAI / Generative AI
try:
    import google.genai  # type: ignore # noqa: F401
    print(f"   [OK]   {'Google GenAI SDK':<35} available")
except ImportError:
    try:
        import google.generativeai  # type: ignore # noqa: F401
        print(f"   [OK]   {'Google GenerativeAI':<35} available")
    except ImportError:
        print(f"   [WARN] {'Google GenAI SDK':<35} not installed (optional)")
        warnings.append("pip install google-genai (for real Gemini API calls)")

# -- 3. Check project files -----------------------------------
print("\n[3] Checking project files in src/...")
required_files = [
    "config.py",
    "data_loader.py",
    "bird_loader.py",
    os.path.join("retrieval", "schema_linker.py"),
    os.path.join("graph", "graph_builder.py"),
    os.path.join("graph", "path_generator.py"),
    os.path.join("graph", "path_scorer.py"),
    os.path.join("graph", "oracle_join_path.py"),
    os.path.join("llm", "sql_generator.py"),
    os.path.join("llm", "llm_factory.py"),
    os.path.join("qce", "complexity_estimator.py"),
    os.path.join("validator", "sql_validator.py"),
    os.path.join("executor", "sql_executor.py"),
    os.path.join("evaluator", "evaluator.py"),
    os.path.join("evaluator", "statistical_test.py"),
]

for f in required_files:
    path = os.path.join(SRC_DIR, f)
    if os.path.exists(path):
        print(f"   [OK]   src/{f}")
    else:
        print(f"   [FAIL] src/{f}  <- MISSING")
        errors.append(f"Missing file: src/{f}")

# -- 4. Check Spider dataset ----------------------------------
print("\n[4] Checking Spider dataset...")
try:
    try:
        from src.config import SPIDER_DIR, BIRD_MINIDEV_DIR, BIRD_DB_DIR
    except ImportError:
        from config import SPIDER_DIR, BIRD_MINIDEV_DIR, BIRD_DB_DIR  # type: ignore

    spider_dir_str = str(SPIDER_DIR)
    spider_files = [
        os.path.join(spider_dir_str, "train_spider.json"),
        os.path.join(spider_dir_str, "dev.json"),
        os.path.join(spider_dir_str, "tables.json"),
        os.path.join(spider_dir_str, "database"),
    ]
    spider_present = True
    for f in spider_files:
        if os.path.exists(f):
            print(f"   [OK]   {os.path.basename(f)}")
        else:
            print(f"   [FAIL] {f}  <- MISSING")
            spider_present = False
            errors.append(f"Spider file missing: {f}")

    # -- 5. Check BIRD Mini-Dev -------------------------------
    print("\n[5] Checking BIRD Mini-Dev dataset...")
    bird_minidev_str = str(BIRD_MINIDEV_DIR)
    bird_db_str = str(BIRD_DB_DIR)
    bird_files = [
        os.path.join(bird_minidev_str, "mini_dev_sqlite.json"),
        os.path.join(bird_minidev_str, "dev_tables.json"),
        bird_db_str,
    ]
    bird_present = True
    for f in bird_files:
        if os.path.exists(f):
            print(f"   [OK]   {os.path.basename(f)}")
        else:
            print(f"   [FAIL] {f}  <- MISSING")
            bird_present = False
            errors.append(f"BIRD file missing: {f}")

except Exception as e:
    print(f"   [FAIL] Could not load config: {e}")
    errors.append("Fix src/config.py")
    spider_present = False
    bird_present = False

# -- 6. Quick SBERT smoke test --------------------------------
print("\n[6] Quick SBERT smoke test...")
try:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    v = m.encode(["test query"], show_progress_bar=False)
    print(f"   [OK]   SBERT encodes correctly (dim={v.shape[1]})")
except Exception as e:
    print(f"   [FAIL] SBERT failed: {e}")
    errors.append(f"sentence-transformers test failed: {e}")

# -- 7. Quick BIRD loader test --------------------------------
print("\n[7] Quick BIRD loader test...")
if bird_present:
    try:
        try:
            from src.bird_loader import BIRDMiniDevLoader
        except ImportError:
            from bird_loader import BIRDMiniDevLoader  # type: ignore

        loader = BIRDMiniDevLoader(str(BIRD_MINIDEV_DIR))
        schemas = loader.load_schemas()
        samples = loader.load_dev()
        print(f"   [OK]   BIRD: {len(schemas)} schemas, {len(samples)} samples loaded")
        if samples:
            s = samples[0]
            print(f"   [OK]   Sample: [{s.db_id}] {s.question[:50]}...")
            print(f"   [OK]   Evidence: {s.evidence[:50] if s.evidence else 'none'}...")
    except Exception as e:
        print(f"   [FAIL] BIRD loader failed: {e}")
        errors.append(f"bird_loader error: {e}")
else:
    print("   [INFO] BIRD files not present; skipping loader test.")

# -- 8. Quick Spider loader test -------------------------------
print("\n[8] Quick Spider loader test...")
if spider_present:
    try:
        try:
            from src.data_loader import SpiderDataLoader
        except ImportError:
            from data_loader import SpiderDataLoader  # type: ignore

        loader = SpiderDataLoader(str(SPIDER_DIR))
        schemas = loader.load_schemas()
        samples = loader.load_train()
        print(f"   [OK]   Spider: {len(schemas)} schemas, {len(samples)} train samples")
    except Exception as e:
        print(f"   [FAIL] Spider loader failed: {e}")
        errors.append(f"Spider loader error: {e}")
else:
    print("   [INFO] Spider files not present; skipping loader test.")

# -- Summary ---------------------------------------------------
print("\n" + "=" * 60)
if not errors:
    print("  [OK] ALL CHECKS PASSED -- ready to run main.py")
    if warnings:
        print("\n  Notices:")
        for w in warnings:
            print(f"     * {w}")
else:
    print(f"  [FAIL] {len(errors)} issue(s) found:")
    for e in errors:
        print(f"     -> {e}")
    print("\n  Fix the above, then run: python check_setup.py again")
print("=" * 60)