# Nearest-neighbour retrieval using cosine similarity to generate recommendations.
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


def load_embedding_table(embeddings_path="movie_embeddings.csv"):
    emb_df = pd.read_csv(embeddings_path)

    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    emb_matrix = emb_df[emb_cols].to_numpy(dtype=np.float32)

    emb_matrix = normalize(emb_matrix, norm="l2").astype(np.float32)

    movie_ids = emb_df["movieId"].to_numpy()
    titles = emb_df["title"].to_numpy()

    movieid_to_idx = {mid: i for i, mid in enumerate(movie_ids)}

    return emb_df, emb_matrix, movieid_to_idx, movie_ids, titles


def build_user_query_vector(
    user_ratings,
    emb_matrix,
    movieid_to_idx,
    weighted=True,
    rating_midpoint=3.0,
    # Legacy parameter kept for call-site compatibility — ignored when weighted=True.
    min_rating=None,
):
    """
    Build a taste query vector using graded, centred weights.

    Every rated movie contributes: those above rating_midpoint attract the
    query toward similar films; those below repel it.  The magnitude of
    influence scales with distance from the midpoint, so a 5★ film pulls
    harder than a 4★ film, and a 1★ film repels harder than a 2.5★ film.

    Parameters
    ----------
    user_ratings : DataFrame with columns ['movieId', 'rating']
    emb_matrix   : (N, D) L2-normalised embedding matrix
    movieid_to_idx : {movieId -> row index in emb_matrix}
    weighted     : if True use centred-rating weights; if False use the mean
                   of embeddings for movies rated above midpoint
    rating_midpoint : the neutral point; ratings above attract, below repel
    min_rating   : deprecated legacy parameter; ignored

    Raises
    ------
    ValueError if no rated movies exist in the embedding table.
    """
    df = user_ratings[user_ratings["movieId"].isin(movieid_to_idx)].copy()
    if df.empty:
        raise ValueError("No rated movies found in the embedding table.")

    idxs = df["movieId"].map(movieid_to_idx).to_numpy()
    vecs = emb_matrix[idxs]

    if weighted:
        # Centre weights: positive → attract, negative → repel
        weights = (df["rating"].to_numpy(dtype=np.float32) - rating_midpoint).reshape(-1, 1)
        query_vec = (vecs * weights).sum(axis=0)
    else:
        # Unweighted fallback: plain mean of above-midpoint movies
        mask = df["rating"].to_numpy() >= rating_midpoint
        query_vec = vecs[mask].mean(axis=0) if mask.any() else vecs.mean(axis=0)

    norm = np.linalg.norm(query_vec)
    if norm < 1e-8:
        # Degenerate case: centred weights cancel out (all ratings near midpoint).
        # Fall back to the mean of above-midpoint embeddings.
        mask = df["rating"].to_numpy() > rating_midpoint
        fallback = vecs[mask] if mask.any() else vecs
        query_vec = fallback.mean(axis=0)
        norm = np.linalg.norm(query_vec)
        if norm < 1e-8:
            raise ValueError("Could not build a meaningful query vector from the given ratings.")

    return (query_vec / norm).astype(np.float32), df["movieId"].tolist()


def recommend_movies_knn(user_ratings, embeddings_path="movie_embeddings.csv",
                         top_k=10, weighted=True, rating_midpoint=3.0):
    """
    Returns top_k movie recommendations using cosine similarity nearest-neighbor retrieval.
    All rated movies contribute via graded centred weights (see build_user_query_vector).
    """
    emb_df, emb_matrix, movieid_to_idx, movie_ids, titles = load_embedding_table(embeddings_path)

    query_vec, _ = build_user_query_vector(
        user_ratings=user_ratings,
        emb_matrix=emb_matrix,
        movieid_to_idx=movieid_to_idx,
        weighted=weighted,
        rating_midpoint=rating_midpoint,
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