import numpy as np
import pandas as pd


def l2_normalize_rows(matrix):
    """Row-wise L2 normalization: divide each row by its L2 norm."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def load_embedding_table(embeddings_path="movie_embeddings.csv"):
    emb_df = pd.read_csv(embeddings_path)

    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    emb_matrix = emb_df[emb_cols].to_numpy(dtype=np.float32)

    emb_matrix = l2_normalize_rows(emb_matrix).astype(np.float32)

    movie_ids = emb_df["movieId"].to_numpy()
    titles = emb_df["title"].to_numpy()

    movieid_to_idx = {mid: i for i, mid in enumerate(movie_ids)}

    return emb_df, emb_matrix, movieid_to_idx, movie_ids, titles


def build_user_query_vector(user_ratings, emb_matrix, movieid_to_idx):
    """
    Build a user query vector using centered rating weights.

    weight = rating - 3.0

    5-star movie -> +2.0
    4-star movie -> +1.0
    3-star movie ->  0.0  
    2-star movie -> -1.0
    1-star movie -> -2.0

    If positive and negative signals cancel (zero norm), falls back to
    liked-only (positive weights) to avoid a crash.
    """
    rated = user_ratings.copy()
    rated = rated[rated["movieId"].isin(movieid_to_idx)].copy()

    if rated.empty:
        raise ValueError("No rated movies found that also exist in the embedding table.")

    rated["weight"] = rated["rating"].astype(np.float32) - 3.0
    rated = rated[rated["weight"] != 0].copy()

    if rated.empty:
        raise ValueError("No usable ratings found after centering around 3.0.")

    idxs = rated["movieId"].map(movieid_to_idx).to_numpy()
    vecs = emb_matrix[idxs]
    weights = rated["weight"].to_numpy(dtype=np.float32).reshape(-1, 1)

    query_vec = (vecs * weights).sum(axis=0)
    norm = np.linalg.norm(query_vec)

    if norm == 0:
        # If positive and negative signals cancelled out.
        # Use only liked movies (positive weights) with no negative signal.
        liked = rated[rated["weight"] > 0]
        if liked.empty:
            raise ValueError("Query vector is zero and no liked movies exist to fall back on.")
        idxs = liked["movieId"].map(movieid_to_idx).to_numpy()
        vecs = emb_matrix[idxs]
        weights = liked["weight"].to_numpy(dtype=np.float32).reshape(-1, 1)
        query_vec = (vecs * weights).sum(axis=0)
        norm = np.linalg.norm(query_vec)

    query_vec = (query_vec / norm).astype(np.float32)
    return query_vec, rated["movieId"].tolist()


def recommend_movies_knn(user_ratings, embeddings_path="movie_embeddings.csv", top_k=10):
    """
    Returns top_k movie recommendations using cosine similarity nearest-neighbor retrieval.
    """
    emb_df, emb_matrix, movieid_to_idx, movie_ids, titles = load_embedding_table(embeddings_path)

    query_vec, used_rated_movies = build_user_query_vector(
        user_ratings=user_ratings,
        emb_matrix=emb_matrix,
        movieid_to_idx=movieid_to_idx
    )

    sims = emb_matrix @ query_vec

    already_rated = set(user_ratings["movieId"].tolist())
    rated_mask = np.array([mid in already_rated for mid in movie_ids])
    sims[rated_mask] = -np.inf

    top_idx = np.argsort(-sims)[:top_k]

    results = pd.DataFrame({
        "movieId": movie_ids[top_idx],
        "title": titles[top_idx],
        "cosine_similarity": sims[top_idx],
    })

    return results.reset_index(drop=True)