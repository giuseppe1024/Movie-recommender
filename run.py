"""
CineMatch launcher — runs the full pipeline then starts the Streamlit app.

Pipeline steps (each skipped if outputs already exist):
  1. Download MovieLens ml-latest-small  (data/movies.csv, data/ratings.csv)
  2. Process raw data                    (data/processed/)       via data/load_data.py
  3. Build movie embeddings              (embeddings/)           via embeddings/build_embeddings.py
  4. Evaluate recommendation quality     (data/eval_results.json) via evaluation/evaluate.py
  5. Launch Streamlit UI                 (app/main.py)

Usage:
    python run.py
"""

import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Key paths ─────────────────────────────────────────────────────────────────
MOVIES_RAW    = ROOT / "data" / "movies.csv"
RATINGS_RAW   = ROOT / "data" / "ratings.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
EMBEDDINGS    = ROOT / "embeddings" / "movie_embeddings.csv"
EVAL_RESULTS  = ROOT / "data" / "eval_results.json"

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


# ── Step 1: Download MovieLens ────────────────────────────────────────────────

def download_movielens():
    """Download and extract the MovieLens ml-latest-small dataset."""
    print("MovieLens data not found.")
    print(f"Downloading from {MOVIELENS_URL} (~1 MB) ...")
    zip_path = ROOT / "data" / "ml-latest-small.zip"
    ROOT / "data"  # ensure parent exists
    urllib.request.urlretrieve(MOVIELENS_URL, str(zip_path))

    print("Extracting ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(ROOT / "data")

    src = ROOT / "data" / "ml-latest-small"
    shutil.copy(src / "movies.csv",  MOVIES_RAW)
    shutil.copy(src / "ratings.csv", RATINGS_RAW)
    shutil.copy(src / "links.csv",   ROOT / "data" / "links.csv")
    shutil.rmtree(src)
    zip_path.unlink(missing_ok=True)
    print("Raw data saved to data/movies.csv and data/ratings.csv\n")


# ── Step 2: Process raw data ──────────────────────────────────────────────────

def process_data():
    """Call load_data() with explicit absolute paths so it works from any cwd."""
    from data.load_data import load_data
    print(">>> Processing raw data ...")
    load_data(
        movies_path=str(MOVIES_RAW),
        ratings_path=str(RATINGS_RAW),
        output_dir=str(PROCESSED_DIR),
    )
    print("Processed data saved to data/processed/\n")


# ── Step 3: Build embeddings ──────────────────────────────────────────────────

def build_embeddings():
    """Call build_embeddings() with explicit absolute paths."""
    from embeddings.build_embeddings import build_embeddings as _build
    print(">>> Building movie embeddings (this may take a few minutes) ...")
    _build(
        processed_dir=str(PROCESSED_DIR),
        output_dir=str(ROOT / "embeddings"),
    )
    print("Embeddings saved to embeddings/movie_embeddings.csv\n")


# ── Step 4: Evaluate recommendations ─────────────────────────────────────────

def run_evaluation():
    """Run temporal leave-k-out evaluation and save summary to data/eval_results.json."""
    from evaluation.evaluate import evaluate_temporal_leave_k_out
    print(">>> Evaluating recommendation quality ...")
    try:
        summary, _ = evaluate_temporal_leave_k_out(
            ratings_path=str(PROCESSED_DIR / "ratings_clean.csv"),
            embeddings_path=str(EMBEDDINGS),
        )
        EVAL_RESULTS.parent.mkdir(parents=True, exist_ok=True)
        EVAL_RESULTS.write_text(json.dumps(summary, indent=2))
        print(f"Evaluation results saved to data/eval_results.json")
        print(f"  Precision@10:  {summary.get('precision_at_k', 0):.4f}")
        print(f"  Recall@10:     {summary.get('recall_at_k', 0):.4f}")
        print(f"  NDCG@10:       {summary.get('ndcg_at_k', 0):.4f}")
        print(f"  Hit Rate@10:   {summary.get('hit_rate_at_k', 0):.4f}\n")
    except Exception as exc:
        print(f"Evaluation skipped ({exc})\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 1. Raw data
    if not (MOVIES_RAW.exists() and RATINGS_RAW.exists()):
        download_movielens()

    # 2. Processed data
    if not (PROCESSED_DIR / "ratings_clean.csv").exists():
        process_data()
    else:
        print("✓ Processed data already exists — skipping.")

    # 3. Embeddings
    if not EMBEDDINGS.exists():
        build_embeddings()
    else:
        print("✓ Embeddings already exist — skipping.")

    # 4. Evaluation (runs once; re-run by deleting data/eval_results.json)
    if not EVAL_RESULTS.exists():
        run_evaluation()
    else:
        print("✓ Evaluation results already exist — skipping.")

    # 5. Launch app
    print("\n>>> Starting CineMatch ...\n")
    subprocess.run(["streamlit", "run", str(ROOT / "app" / "main.py")])


if __name__ == "__main__":
    main()
