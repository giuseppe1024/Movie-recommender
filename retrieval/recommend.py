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


def build_user_query_vector(
    user_ratings,
    emb_matrix,
    movieid_to_idx,
    weighted=True,
    rating_midpoint=3.0
):
    """
    Build a user query vector using centered rating weights.

    weight = rating - 3.0

    5-star movie -> +2.0
    4-star movie -> +1.0
    3-star movie ->  0.0
    2-star movie -> -1.0
    1-star movie -> -2.0
    """
    rated = user_ratings.copy()
    rated = rated[rated["movieId"].isin(movieid_to_idx)].copy()

    if rated.empty:
        raise ValueError("No rated movies found that also exist in the embedding table.")

    if weighted:
        rated["weight"] = rated["rating"].astype(np.float32) - rating_midpoint
    else:
        rated["weight"] = 1.0
    rated = rated[rated["weight"] != 0].copy()

    if rated.empty:
        raise ValueError("No usable ratings found after centering around 3.0.")

    idxs = rated["movieId"].map(movieid_to_idx).to_numpy()
    vecs = emb_matrix[idxs]
    weights = rated["weight"].to_numpy(dtype=np.float32).reshape(-1, 1)

    query_vec = (vecs * weights).sum(axis=0)
    norm = np.linalg.norm(query_vec)

    if norm == 0:
        liked = rated[rated["weight"] > 0].copy()

        if liked.empty:
            raise ValueError("Query vector is zero and no liked movies exist to fall back on.")

        idxs = liked["movieId"].map(movieid_to_idx).to_numpy()
        vecs = emb_matrix[idxs]
        weights = liked["weight"].to_numpy(dtype=np.float32).reshape(-1, 1)

        query_vec = (vecs * weights).sum(axis=0)
        norm = np.linalg.norm(query_vec)

    query_vec = (query_vec / norm).astype(np.float32)

    return query_vec, rated["movieId"].tolist()


def split_genres(genre_value):
    """
    Convert a MovieLens-style genre string into a set.

    Example:
    "Action|Adventure|Sci-Fi" -> {"Action", "Adventure", "Sci-Fi"}
    """
    if pd.isna(genre_value):
        return set()

    genre_value = str(genre_value)

    if genre_value == "(no genres listed)":
        return set()

    return set(g.strip() for g in genre_value.split("|") if g.strip())


def load_movie_metadata(movies_path=None, emb_df=None):
    """
    Loads movie metadata for explanation features.

    If movies_path is given, it expects a file with at least:
        movieId, title, genres

    If movies_path is None, it tries to use columns already inside emb_df.
    """
    if movies_path is not None:
        movies = pd.read_csv(movies_path)
    else:
        movies = emb_df.copy()

    if "movieId" not in movies.columns:
        raise ValueError("Movie metadata must contain a movieId column.")

    if "title" not in movies.columns:
        raise ValueError("Movie metadata must contain a title column.")

    if "genres" not in movies.columns:
        movies["genres"] = ""

    movie_info = movies[["movieId", "title", "genres"]].drop_duplicates("movieId").copy()
    return movie_info


def build_popularity_table(ratings_path=None):
    """
    Builds popularity statistics from the full ratings dataset.

    Returns:
        movieId
        rating_count
        avg_rating
    """
    if ratings_path is None:
        return None

    ratings = pd.read_csv(ratings_path)

    popularity = (
        ratings
        .groupby("movieId")
        .agg(
            rating_count=("rating", "count"),
            avg_rating=("rating", "mean")
        )
        .reset_index()
    )

    return popularity


def build_coliked_pairs(ratings_path=None, min_rating=4.0):
    """
    Builds a dictionary for co-liked movie pairs.

    If two movies are both rated highly by the same user, their pair count increases.

    This supports the explanation:
        "Fans of X also enjoyed this"
    """
    if ratings_path is None:
        return {}

    ratings = pd.read_csv(ratings_path)
    high = ratings[ratings["rating"] >= min_rating].copy()

    user_movies = high.groupby("userId")["movieId"].apply(list)

    pair_counts = {}

    for movies in user_movies:
        movies = list(set(movies))

        for i in range(len(movies)):
            for j in range(i + 1, len(movies)):
                a, b = sorted([movies[i], movies[j]])
                pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

    return pair_counts


def get_pair_count(movie_a, movie_b, pair_counts):
    a, b = sorted([movie_a, movie_b])
    return pair_counts.get((a, b), 0)


def explain_similar_to(
    rec_movie_id,
    liked_movie_ids,
    movieid_to_title,
    pair_counts,
    emb_matrix,
    movieid_to_idx
):
    """
    Finds the best anchor movie from the user's liked history.

    First tries strongest co-liked signal from training ratings.
    If no co-liked signal exists, falls back to embedding similarity.
    """
    if len(liked_movie_ids) == 0:
        return ""

    best_movie = None
    best_count = -1

    for liked_id in liked_movie_ids:
        count = get_pair_count(rec_movie_id, liked_id, pair_counts)

        if count > best_count:
            best_count = count
            best_movie = liked_id

    if best_count > 0:
        return movieid_to_title.get(best_movie, "")

    # Fallback: use embedding similarity to find nearest liked movie.
    if rec_movie_id not in movieid_to_idx:
        return ""

    rec_vec = emb_matrix[movieid_to_idx[rec_movie_id]]

    best_movie = None
    best_sim = -np.inf

    for liked_id in liked_movie_ids:
        if liked_id not in movieid_to_idx:
            continue

        liked_vec = emb_matrix[movieid_to_idx[liked_id]]
        sim = float(rec_vec @ liked_vec)

        if sim > best_sim:
            best_sim = sim
            best_movie = liked_id

    if best_movie is None:
        return ""

    return movieid_to_title.get(best_movie, "")


def explain_shared_genres(rec_movie_id, liked_movie_ids, movieid_to_genres):
    """
    Finds genres shared between the recommended movie and the user's liked movies.
    """
    rec_genres = movieid_to_genres.get(rec_movie_id, set())

    if not rec_genres:
        return ""

    liked_genres = set()

    for liked_id in liked_movie_ids:
        liked_genres.update(movieid_to_genres.get(liked_id, set()))

    shared = rec_genres.intersection(liked_genres)

    if not shared:
        return ""

    return ", ".join(sorted(shared))


def explain_popularity(movie_id, popularity_dict):
    """
    Creates a readable popularity explanation.
    """
    if popularity_dict is None or movie_id not in popularity_dict:
        return ""

    rating_count, avg_rating = popularity_dict[movie_id]

    return f"{int(rating_count)} ratings, average {avg_rating:.2f}"


def add_recommendation_explanations(
    results,
    user_ratings,
    movie_info,
    ratings_path,
    emb_matrix,
    movieid_to_idx
):
    """
    Adds Slide 6 explanation features:

    1. similar_to:
       A liked movie from the user's history that anchors the recommendation.

    2. shared_genres:
       Genres shared between the recommendation and movies the user liked.

    3. popularity:
       Number of ratings and average rating from the full ratings dataset.
    """
    movieid_to_title = dict(zip(movie_info["movieId"], movie_info["title"]))

    movieid_to_genres = {
        row["movieId"]: split_genres(row["genres"])
        for _, row in movie_info.iterrows()
    }

    liked_movie_ids = (
        user_ratings[user_ratings["rating"] >= 4.0]["movieId"]
        .dropna()
        .tolist()
    )

    pair_counts = build_coliked_pairs(ratings_path=ratings_path)

    popularity_table = build_popularity_table(ratings_path=ratings_path)

    if popularity_table is not None:
        popularity_dict = {
            row["movieId"]: (row["rating_count"], row["avg_rating"])
            for _, row in popularity_table.iterrows()
        }
    else:
        popularity_dict = None

    similar_to_values = []
    shared_genres_values = []
    popularity_values = []

    for movie_id in results["movieId"]:
        similar_to = explain_similar_to(
            rec_movie_id=movie_id,
            liked_movie_ids=liked_movie_ids,
            movieid_to_title=movieid_to_title,
            pair_counts=pair_counts,
            emb_matrix=emb_matrix,
            movieid_to_idx=movieid_to_idx
        )

        shared_genres = explain_shared_genres(
            rec_movie_id=movie_id,
            liked_movie_ids=liked_movie_ids,
            movieid_to_genres=movieid_to_genres
        )

        popularity = explain_popularity(
            movie_id=movie_id,
            popularity_dict=popularity_dict
        )

        similar_to_values.append(similar_to)
        shared_genres_values.append(shared_genres)
        popularity_values.append(popularity)

    results = results.copy()
    results["similar_to"] = similar_to_values
    results["shared_genres"] = shared_genres_values
    results["popularity"] = popularity_values

    return results


def recommend_movies_knn(
    user_ratings,
    embeddings_path="movie_embeddings.csv",
    movies_path=None,
    ratings_path=None,
    top_k=10
):
    """
    Returns top_k movie recommendations using cosine similarity nearest-neighbor retrieval.

    Also adds explanation features:
        similar_to
        shared_genres
        popularity
    """
    emb_df, emb_matrix, movieid_to_idx, movie_ids, titles = load_embedding_table(
        embeddings_path=embeddings_path
    )

    movie_info = load_movie_metadata(
        movies_path=movies_path,
        emb_df=emb_df
    )

    query_vec, used_rated_movies = build_user_query_vector(
        user_ratings=user_ratings,
        emb_matrix=emb_matrix,
        movieid_to_idx=movieid_to_idx
    )

    # Since both movie embeddings and query vector are L2-normalized,
    # dot product equals cosine similarity.
    sims = emb_matrix @ query_vec

    # Mask out movies the user already rated.
    already_rated = set(user_ratings["movieId"].tolist())
    rated_mask = np.array([mid in already_rated for mid in movie_ids])
    sims[rated_mask] = -np.inf

    top_idx = np.argsort(-sims)[:top_k]

    results = pd.DataFrame({
        "movieId": movie_ids[top_idx],
        "title": titles[top_idx],
        "cosine_similarity": sims[top_idx]
    })

    results = add_recommendation_explanations(
        results=results,
        user_ratings=user_ratings,
        movie_info=movie_info,
        ratings_path=ratings_path,
        emb_matrix=emb_matrix,
        movieid_to_idx=movieid_to_idx
    )

    return results.reset_index(drop=True)