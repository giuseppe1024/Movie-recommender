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