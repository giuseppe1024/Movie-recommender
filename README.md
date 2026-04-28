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

The simplest way to run the full pipeline (downloads data, processes it, builds embeddings, runs evaluation, then launches the app):

```bash
python run.py
```

Each step is skipped automatically if its outputs already exist. To force a fresh evaluation, delete `data/eval_results.json` before running.

To launch the app directly (if data and embeddings are already built):

```bash
streamlit run app/main.py
```

### Pipeline steps
1. **Download** — fetches MovieLens ml-latest-small (~1 MB) into `data/`
2. **Process** — cleans and splits raw data into `data/processed/`
3. **Embeddings** — builds movie embeddings into `embeddings/`
4. **Evaluate** — runs sliding-window leave-k-out evaluation, saves metrics to `data/eval_results.json`
5. **Launch** — starts the Streamlit UI at `http://localhost:8501`

## Data Setup

This project uses the MovieLens Latest Small dataset that contains 100,000 movie ratings from 610 users across 9,742 different movies.

### How to get the data
1. Go to https://grouplens.org/datasets/movielens/latest/
2. Download **ml-latest-small.zip**
3. Unzip it and copy **movies.csv** and **ratings.csv** into the `data/` folder
4. Run the following command to generate all cleaned data files:

python data/load_data.py

It will create a processed/ folder inside data/ with the following:
a) `clean_movies.csv` --> cleaned movie metadata
b) `ratings_clean.csv` —-> cleaned user ratings
c) `movie_stats.csv` —-> per-movie rating count and mean
d) `embedding_movies.csv` —-> ready for the embedding process
e) `movies_with_stats.csv` —-> movies merged with their stats
