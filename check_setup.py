# check_setup.py
# ============================================================
# Run this FIRST to verify all packages and files are correct
# Usage: python check_setup.py
# ============================================================

import sys, os
print("="*60)
print("  ConSQL Setup Checker")
print("="*60)

errors = []
warnings = []

# ── 1. Check Python version ──────────────────────────────────
print(f"\n[1] Python version: {sys.version.split()[0]}")

# ── 2. Check all packages ────────────────────────────────────
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
    "tqdm":                  "TQDM",
    "gradio":                "Gradio",
    "accelerate":            "Accelerate",
}
for pkg, name in packages.items():
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "?")
        print(f"   ✅ {name:<35} v{ver}")
    except ImportError:
        print(f"   ❌ {name:<35} NOT FOUND")
        errors.append(f"pip install {pkg}")

# ── 3. Check project files ───────────────────────────────────
print("\n[3] Checking project files in src/...")
src_dir = os.path.join(os.path.dirname(__file__), "src")
required_files = [
    "config.py", "data_loader.py", "schema_linker.py",
    "graph_builder.py", "path_generator.py", "path_scorer.py",
    "sql_generator.py", "evaluator.py", "bird_loader.py"
]
for f in required_files:
    path = os.path.join(src_dir, f)
    if os.path.exists(path):
        print(f"   ✅ src/{f}")
    else:
        print(f"   ❌ src/{f}  ← MISSING")
        errors.append(f"Copy {f} to src/")

# ── 4. Check Spider dataset ──────────────────────────────────
print("\n[4] Checking Spider dataset...")
sys.path.insert(0, src_dir)
try:
    from config import SPIDER_DIR, BIRD_MINIDEV_DIR, BIRD_DB_DIR
    spider_files = [
        os.path.join(SPIDER_DIR, "train_spider.json"),
        os.path.join(SPIDER_DIR, "dev.json"),
        os.path.join(SPIDER_DIR, "tables.json"),
        os.path.join(SPIDER_DIR, "database"),
    ]
    for f in spider_files:
        if os.path.exists(f):
            print(f"   ✅ {os.path.basename(f)}")
        else:
            print(f"   ❌ {f}  ← MISSING")
            errors.append(f"Spider file missing: {f}")

    # ── 5. Check BIRD Mini-Dev ───────────────────────────────
    print("\n[5] Checking BIRD Mini-Dev dataset...")
    bird_files = [
        os.path.join(BIRD_MINIDEV_DIR, "mini_dev_sqlite.json"),
        os.path.join(BIRD_MINIDEV_DIR, "dev_tables.json"),
        BIRD_DB_DIR,
    ]
    for f in bird_files:
        if os.path.exists(f):
            print(f"   ✅ {os.path.basename(f)}")
        else:
            print(f"   ❌ {f}  ← MISSING")
            errors.append(f"BIRD file missing: {f}")

except Exception as e:
    print(f"   ❌ Could not load config: {e}")
    errors.append("Fix src/config.py")

# ── 6. Quick SBERT smoke test ────────────────────────────────
print("\n[6] Quick SBERT smoke test...")
try:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    v = m.encode(["test query"], show_progress_bar=False)
    print(f"   ✅ SBERT encodes correctly (dim={v.shape[1]})")
except Exception as e:
    print(f"   ❌ SBERT failed: {e}")
    errors.append("sentence-transformers broken")

# ── 7. Quick BIRD loader test ────────────────────────────────
print("\n[7] Quick BIRD loader test...")
try:
    from bird_loader import BIRDMiniDevLoader
    from config import BIRD_MINIDEV_DIR
    loader = BIRDMiniDevLoader(BIRD_MINIDEV_DIR)
    schemas = loader.load_schemas()
    samples = loader.load_dev()
    print(f"   ✅ BIRD: {len(schemas)} schemas, {len(samples)} samples loaded")
    if samples:
        s = samples[0]
        print(f"   ✅ Sample: [{s.db_id}] {s.question[:50]}...")
        print(f"   ✅ Evidence: {s.evidence[:50] if s.evidence else 'none'}...")
except Exception as e:
    print(f"   ❌ BIRD loader failed: {e}")
    errors.append(f"bird_loader error: {e}")

# ── 8. Quick Spider loader test ───────────────────────────────
print("\n[8] Quick Spider loader test...")
try:
    from data_loader import SpiderDataLoader
    from config import SPIDER_DIR
    loader = SpiderDataLoader(SPIDER_DIR)
    schemas = loader.load_schemas()
    samples = loader.load_train()
    print(f"   ✅ Spider: {len(schemas)} schemas, {len(samples)} train samples")
except Exception as e:
    print(f"   ❌ Spider loader failed: {e}")
    errors.append(f"Spider loader error: {e}")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "="*60)
if not errors:
    print("  ✅ ALL CHECKS PASSED — ready to run main.py")
else:
    print(f"  ❌ {len(errors)} issue(s) found:")
    for e in errors:
        print(f"     → {e}")
    print("\n  Fix the above, then run: python check_setup.py again")
print("="*60)