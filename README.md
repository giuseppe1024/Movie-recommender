# CineMatch — Movie Recommender

A Streamlit web app that recommends movies based on your personal rating history, using semantically-aware embeddings, contrastive fine-tuning, and nearest-neighbour retrieval.

**Team:** Vayun Malik · Kathy Lin · Kardelen Kalyon · Linyi Huang · Giuseppe Aprile Borriello

---

## How It Works

### 1. Embeddings (`embeddings/build_embeddings.py`)
- Each movie's title, genres, and user-generated tags are concatenated into a single text string.
- A pre-trained `all-MiniLM-L6-v2` sentence transformer encodes each movie into a 384-dim vector.
- A custom 2-layer MLP projection head (384→256→128) is fine-tuned using **NT-Xent contrastive loss** (implemented from scratch in PyTorch/NumPy — no library call computes the loss).
- **Positive pairs:** movies co-rated ≥ 4★ by ≥ 10 users are pulled together in embedding space.
- Final 128-dim vectors are L2-normalised so cosine similarity reduces to a dot product.

### 2. Retrieval (`retrieval/recommend.py`)
- A **user query vector** is built as a weighted centroid of the user's rated-movie embeddings.
- Weights are **centered:** `weight = rating − 3.0`, so 5★ films attract (+2), 3★ films are neutral (0), and 1★ films actively repel (−2). This is implemented in pure NumPy.
- Top-K recommendations are found by dot-product similarity over all ~9K movies (O(n), instant).
- Already-rated movies are masked to −∞ before ranking.

### 3. Evaluation (`evaluation/evaluate.py`)
- **Temporal leave-k-out** with sliding windows: holds out a user's k most recent ratings as the test set, trains only on earlier ones (no future leakage).
- All metrics are implemented from scratch in NumPy:
  - **Hit Rate@K** — fraction of users who got ≥ 1 relevant recommendation
  - **Dislike Rate@K** — fraction of verified recommendations that were actively disliked (≤ threshold)
  - **Pairwise Rank Accuracy** — does the model rank a 5★ movie above a 4★ movie?
  - **Precision@K**, **Binary NDCG@K**, **Graded NDCG@K**, **Catalog Coverage**

### 4. App (`app/main.py`)
- Multi-page Streamlit app with a dark cinema-themed UI.
- **Browse** — explore 9,000+ films, filter by genre, decade, and rating.
- **For You** — personalised top-K recommendations with three per-movie reason bullets: most similar liked movie, shared genres, and community signal.
- **My Ratings** — your rated collection with sort/filter, plus a **Taste Profile** subsection:
  - Genre bar chart (liked vs. disliked, weighted by distance from 3★)
  - Interactive 3D PCA of the embedding space with your rated movies and query vector plotted
- **Evaluation** — live sensitivity analysis and full metric breakdown with explanations.

---

## Project Structure

```
app/               Streamlit web application (main.py)
data/              Raw and processed data, evaluation results
embeddings/        Embedding pipeline (build_embeddings.py, movie_embeddings.csv)
evaluation/        Evaluation pipeline (evaluate.py)
retrieval/         Nearest-neighbour retrieval (recommend.py)
run.py             Full pipeline orchestrator
requirements.txt   Python dependencies
```

---

## Setup & Running

**Requires Python 3.10+**
**Requires Streamlit 1.57+**

### Quickest start (data and embeddings already in repo)
```bash
pip install -r requirements.txt
streamlit run app/main.py
```

### Full pipeline (re-download and regenerate everything)
```bash
pip install -r requirements.txt
python run.py
```

`run.py` downloads MovieLens ml-latest-small, processes data, builds embeddings, runs evaluation, and launches the app. Each step is skipped if its outputs already exist.

### Force a fresh evaluation
```bash
rm data/eval_results.json
python run.py
```

---

## Dataset

**MovieLens ml-latest-small** — 100,836 ratings from 610 users across 9,742 movies. Includes user-generated tag data used to enrich movie text for embedding.

Processed files are written to `data/processed/`:
| File | Contents |
|---|---|
| `clean_movies.csv` | Deduplicated movie metadata with cleaned titles/genres |
| `ratings_clean.csv` | Filtered ratings (duplicates removed, types cast) |
| `movie_stats.csv` | Per-movie mean rating and rating count |
| `embedding_movies.csv` | Text field ready for sentence-transformer encoding |
| `movies_with_stats.csv` | Movies joined with statistics |

---

## Dependencies

See `requirements.txt`. Key packages:
- `streamlit` — web UI
- `sentence-transformers` — base BERT encoder
- `torch` — contrastive fine-tuning
- `scikit-learn` — PCA for visualisation only
- `plotly`, `matplotlib` — charts
- `numpy`, `pandas` — all retrieval and evaluation logic
