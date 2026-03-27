import re
import pandas as pd
from pathlib import Path

def year_extraction(movie_titles):
    if pd.isna(movie_titles):
        return None
    match = re.search(r"\((\d{4})\)\s*$", str(movie_titles))
    return int(match.group(1)) if match else None


def titles_clean(movie_titles):
    if pd.isna(movie_titles):
        return ""
    movie_titles = str(movie_titles).strip()
    movie_titles = re.sub(r"\s*\(\d{4}\)\s*$", "", movie_titles)
    return movie_titles.lower().strip()


def genres_clean(genre):
    if pd.isna(genre) or genre == "(no genres listed)":
        return "unknown"
    return str(genre).replace("|", " ").lower().strip()


# Cleaning 
def clean_movies(movies):
    movies = movies.copy()
    movies = movies.drop_duplicates(subset=["movieId"])
    movies["year"] = movies["title"].apply(year_extraction)
    movies["clean_titles"] = movies["title"].apply(titles_clean)
    movies["clean_genres"] = movies["genres"].apply(genres_clean)
    movies["movies_text"] = (
        movies["clean_titles"].fillna("") + " " + movies["clean_genres"].fillna("")
    ).str.strip()
    return movies


def clean_ratings(ratings):
    ratings = ratings.copy()
    ratings = ratings.dropna(subset=["userId", "movieId", "rating"])
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)
    if "timestamp" in ratings.columns:
        ratings = ratings.sort_values("timestamp")
        ratings = ratings.drop_duplicates(subset=["userId", "movieId"], keep="last")
    return ratings


def movies_filter(ratings_df, movies_df):
    valid_ids = set(movies_df["movieId"].unique())
    return ratings_df[ratings_df["movieId"].isin(valid_ids)].copy()


def create_stats(ratings_df):
    return (
        ratings_df.groupby("movieId")["rating"]
        .agg(count="count", mean_rating="mean")
        .reset_index()
    )


def apply_filters(clean_df, clean_ratings_df, movie_stats, min_ratings=5):
    # Filter by minimum ratings
    popular_movie_ids = movie_stats.loc[
        movie_stats["count"] >= min_ratings, "movieId"
    ]
    clean_df_filter = clean_df[clean_df["movieId"].isin(popular_movie_ids)].copy()
    clean_ratings_filter = clean_ratings_df[
        clean_ratings_df["movieId"].isin(popular_movie_ids)
    ].copy()
    movie_stats_filter = movie_stats[
        movie_stats["movieId"].isin(popular_movie_ids)
    ].copy()

    # Fill all the missing years and drop unknown genres
    clean_df_filter["year"] = clean_df_filter["year"].fillna(-1).astype(int)
    clean_df_filter = clean_df_filter[
        clean_df_filter["clean_genres"] != "unknown"
    ].copy()

    # Re-arrange all ratings and stats to match final movie list
    valid_movies = set(clean_df_filter["movieId"])
    clean_ratings_filter = clean_ratings_filter[
        clean_ratings_filter["movieId"].isin(valid_movies)
    ].copy()
    movie_stats_filter = movie_stats_filter[
        movie_stats_filter["movieId"].isin(valid_movies)
    ].copy()

    return clean_df_filter, clean_ratings_filter, movie_stats_filter


# Main
def load_data(movies_path="movies.csv", ratings_path="ratings.csv",
              output_dir="processed", min_ratings=5):

    # Load
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)

    # Clean
    clean_df = clean_movies(movies)
    clean_ratings_df = clean_ratings(ratings)
    clean_ratings_df = movies_filter(clean_ratings_df, clean_df)

    # Stats
    movie_stats = create_stats(clean_ratings_df)

    # Filter all 
    clean_df_filter, clean_ratings_filter, movie_stats_filter = apply_filters(
        clean_df, clean_ratings_df, movie_stats, min_ratings=min_ratings
    )

    # Create output data 
    movies_embedding = clean_df_filter[
        ["movieId", "title", "clean_titles", "year", "genres", "movies_text"]
    ].copy()
    movies_with_stats = clean_df_filter.merge(
        movie_stats_filter, on="movieId", how="left"
    )

    # Overall Check 
    print("clean_df_filter shape:", clean_df_filter.shape)
    print("clean_ratings_filter shape:", clean_ratings_filter.shape)
    print("movie_stats_filter shape:", movie_stats_filter.shape)
    print("Unique movies:", clean_df_filter["movieId"].nunique())
    print("Unique rated movies:", clean_ratings_filter["movieId"].nunique())
    print("Unique users:", clean_ratings_filter["userId"].nunique())
    print("Missing years:", clean_df_filter["year"].isna().sum())
    print("Unknown genres:", (clean_df_filter["clean_genres"] == "unknown").sum())
    print("Rating range:", clean_ratings_filter["rating"].min(),
          clean_ratings_filter["rating"].max())

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    clean_df_filter.to_csv(output_dir / "clean_movies.csv", index=False)
    clean_ratings_filter.to_csv(output_dir / "ratings_clean.csv", index=False)
    movie_stats_filter.to_csv(output_dir / "movie_stats.csv", index=False)
    movies_embedding.to_csv(output_dir / "embedding_movies.csv", index=False)
    movies_with_stats.to_csv(output_dir / "movies_with_stats.csv", index=False)
    print("Saved all processed files.")

    return clean_df_filter, clean_ratings_filter, movie_stats_filter, \
           movies_embedding, movies_with_stats


if __name__ == "__main__":
    load_data()
