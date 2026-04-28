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


def build_user_query_vector(user_ratings, emb_matrix, movieid_to_idx, min_rating=4.0,
                            weighted=True):
    """
    user_ratings: DataFrame with columns ['movieId', 'rating']
    min_rating: only use movies rated at least this highly
    weighted: if True, weight each movie embedding by its rating
    """
    liked = user_ratings[user_ratings["rating"] >= min_rating].copy()

    valid_rows = liked["movieId"].isin(movieid_to_idx)
    liked = liked[valid_rows]

    if liked.empty:
        raise ValueError("No highly-rated movies found that also exist in the embedding table.")

    idxs = liked["movieId"].map(movieid_to_idx).to_numpy()
    vecs = emb_matrix[idxs]

    if weighted:
        weights = liked["rating"].to_numpy(dtype=np.float32).reshape(-1, 1)
        query_vec = (vecs * weights).sum(axis=0) / weights.sum()
    else:
        query_vec = vecs.mean(axis=0)

    # Normalize so cosine similarity = dot product
    query_vec = query_vec.astype(np.float32)
    query_vec = query_vec / np.linalg.norm(query_vec)

    return query_vec, liked["movieId"].tolist()


def recommend_movies_knn(user_ratings, embeddings_path="movie_embeddings.csv",
                         min_rating=4.0, top_k=10, weighted=True):
    """
    Returns top_k movie recommendations using cosine similarity nearest-neighbor retrieval
    """
    emb_df, emb_matrix, movieid_to_idx, movie_ids, titles = load_embedding_table(embeddings_path)

    query_vec, seen_high_rated = build_user_query_vector(
        user_ratings=user_ratings,
        emb_matrix=emb_matrix,
        movieid_to_idx=movieid_to_idx,
        min_rating=min_rating,
        weighted=weighted
    )

    sims = emb_matrix @ query_vec

    already_rated = set(user_ratings["movieId"].tolist())
    rated_mask = np.array([mid in already_rated for mid in movie_ids])
    sims[rated_mask] = -np.inf

    top_idx = np.argsort(-sims)[:top_k]

    results = pd.DataFrame({
        "movieId": movie_ids[top_idx],
        "title": titles[top_idx],
        "cosine_similarity": sims[top_idx]
    })

    return results.reset_index(drop=True)