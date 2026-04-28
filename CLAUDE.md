# CineMatch — Claude Project Context

## Project Overview
CineMatch is a Streamlit-based movie recommendation web app built for NYU's Foundations of ML course (Spring 2026). It recommends movies using semantic embeddings and nearest-neighbour retrieval, with a full evaluation pipeline.

**Team:** Vayun Malik, Kathy Lin, Kardelen Kalyon, Linyi Huang, Giuseppe Aprile Borriello

## Stack
- **Frontend:** Streamlit (`app/main.py`) — single-file multi-page app
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`) + contrastive fine-tuning (`embeddings/build_embeddings.py`)
- **Retrieval:** Cosine similarity nearest-neighbour via dot product on L2-normalised vectors (`retrieval/recommend.py`)
- **Evaluation:** Temporal sliding-window leave-k-out (`evaluation/evaluate.py`)
- **Data:** MovieLens dataset; processed files in `data/processed/`

## Running
```bash
python run.py          # full pipeline: download → process → embed → evaluate → launch app
streamlit run app/main.py   # app only (requires embeddings to exist)
```

## Architecture Notes

### Embeddings (`embeddings/build_embeddings.py`)
1. Encode each movie's text (title + genres + tags) with `all-MiniLM-L6-v2` → 384-dim base vectors
2. Build contrastive pairs: movies co-rated ≥4★ by ≥10 users are treated as positives
3. Fine-tune a 2-layer MLP projection head (384→256→128) using NT-Xent loss
4. L2-normalise final 128-dim vectors so cosine similarity = dot product

### Retrieval (`retrieval/recommend.py`)
- Build a query vector: weighted average of the user's liked-movie embeddings (weight = rating)
- Rank all movies by dot product with the query vector (cosine sim)
- Mask out already-rated movies; return top-K

### Evaluation (`evaluation/evaluate.py`)
- Temporal leave-k-out with sliding windows (up to 3 windows per user)
- Metrics: Precision@K, Binary NDCG@K, Graded NDCG@K, Hit Rate@K, Pairwise Rank Accuracy, Dislike Rate@K, Catalog Coverage

## UI Decisions
- Sensitivity analysis on the Evaluation page only exposes **K (movies held out)** and **Liked threshold** as sweep parameters. `top_k` (N) and `min_relevant_test` were removed as they are not of interest for sensitivity analysis.
- Evaluation sidebar lets users set N, K, relevance threshold, and max windows, then re-run evaluation.
- Poster images: local pre-cached CSV first, TMDB API fallback, gradient placeholder last.
