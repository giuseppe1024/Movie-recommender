# Movie Recommender

A web app that recommends movies based on your personal rating history,
using learned embeddings and nearest neighbour retrieval.

## Team
- Vayun Malik
- Kathy Lin
- Kardelen Kalyon
- Linyi Huang
- Giuseppe Aprile Borriello

## Project Structure
- `data/` — data loading and preprocessing
- `embeddings/` — building and storing movie embeddings
- `retrieval/` — nearest neighbour search and recommendation logic
- `app/` — Streamlit web application
- `evaluation/` — model evaluation and validation

## Setup
```bash
pip install -r requirements.txt
```

## Running the app
```bash
streamlit run app/main.py
```

## Data Setup

This project uses the MovieLens Latest Small dataset.

### How to get the data
1. Go to https://grouplens.org/datasets/movielens/latest/
2. Download **ml-latest-small.zip**
3. Unzip it and copy **movies.csv** and **ratings.csv** into the `data/` folder
4. Run the following command to generate all cleaned files:

python data/load_data.py

It will create a `processed/` folder inside `data/` with the following:
- `clean_movies.csv` — cleaned movie metadata
- `ratings_clean.csv` — cleaned user ratings
- `movie_stats.csv` — per-movie rating count and mean
- `embedding_movies.csv` — ready for Vayun's embedding step
- `movies_with_stats.csv` — movies merged with their stats
