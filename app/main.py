"""CineMatch — Movie recommendation Streamlit app."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.recommend import build_user_query_vector, load_embedding_table

# ── Paths ─────────────────────────────────────────────────────────────────────
EMBEDDINGS_PATH = PROJECT_ROOT / "embeddings" / "movie_embeddings.csv"
MOVIES_PATH     = PROJECT_ROOT / "data" / "processed" / "embedding_movies.csv"
STATS_PATH      = PROJECT_ROOT / "data" / "processed" / "movie_stats.csv"
OMDB_CACHE_PATH   = PROJECT_ROOT / "data" / "omdb_cache.json"
EVAL_RESULTS_PATH = PROJECT_ROOT / "data" / "eval_results.json"
LINKS_PATH        = PROJECT_ROOT / "data" / "links.csv"
POSTER_MAPPING_PATH = PROJECT_ROOT / "data" / "processed" / "poster_mapping.csv"

RATING_OPTIONS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
PAGE_SIZE = 20

POSTER_GRADIENTS = [
    ("#c0392b", "#5a0000"), ("#1a6eb5", "#082040"), ("#c9a227", "#4a3800"),
    ("#1a7a40", "#062010"), ("#7b2fbe", "#220040"), ("#0a7a70", "#011818"),
    ("#c45e0a", "#3a1800"), ("#8b2252", "#280010"), ("#2c7873", "#081818"),
    ("#6a3520", "#200800"),
]

# ── Articles to move from suffix to prefix ────────────────────────────────────
_ARTICLE_RE = re.compile(
    r"^(.*?),\s+(The|A|An|La|Le|Les|Los|Las|El|L')\s*((?:\([^)]*\)\s*)*)(\(\d{4}\))?\s*$",
    re.IGNORECASE,
)

def format_display_title(raw: str) -> str:
    """
    Convert MovieLens title conventions to clean display titles.
      'American President, The (1995)'       → 'The American President (1995)'
      "'burbs, The (1989)"                   → "The 'Burbs (1989)"
      '*batteries not included (1987)'       → 'Batteries Not Included (1987)'
      'City of Lost Children, The (Cité...) (1995)' → 'The City of Lost Children (Cité...) (1995)'
    """
    t = raw.strip()

    # Strip leading * (e.g. *batteries not included)
    stripped_star = t.lstrip("*").strip()
    was_star = stripped_star != t
    t = stripped_star

    # Fix "Title, The (optional foreign) (year)" → "The Title (optional foreign) (year)"
    m = _ARTICLE_RE.match(t)
    if m:
        main    = m.group(1).strip()
        article = m.group(2).strip()
        foreign = m.group(3).strip()   # e.g. "(Cité des enfants perdus, La)"
        year    = m.group(4) or ""
        parts   = [article, main]
        if foreign:
            parts.append(foreign)
        if year:
            parts.append(year)
        t = " ".join(parts)
    elif was_star:
        # Title case for things like "batteries not included"
        minor = {"a","an","the","and","but","or","nor","at","by","for",
                 "in","of","on","to","up","as","is"}
        words = t.split()
        t = " ".join(
            w if w.lower() in minor and i > 0 else w.capitalize()
            for i, w in enumerate(words)
        )

    return t

def sort_key(display_title: str) -> str:
    """Strip leading articles for alphabetical sort (matches library convention)."""
    t = display_title.lower()
    for art in ("the ", "a ", "an "):
        if t.startswith(art):
            return t[len(art):]
    return t


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch", page_icon="🎬",
    layout="wide", initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700;900&family=Inter:wght@300;400;500;600&display=swap');

.stApp { background: #0a0a0a; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #e8e8e8; }

/* ── Sidebar: wider, deeper shadow, clearly distinct from main content ── */
section[data-testid="stSidebar"] { min-width: 290px !important; max-width: 340px !important; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111111 0%, #0d0d0d 100%) !important;
    border-right: 2px solid #252525 !important;
    box-shadow: 6px 0 30px rgba(0,0,0,0.6) !important;
}
[data-testid="stSidebar"] * { color: #d8d8d8 !important; }

/* ── Nav links: larger, clearly interactive, with hover/active states ── */
[data-testid="stSidebarNavLink"] {
    border-radius: 8px !important;
    margin: 3px 6px !important;
    padding: 11px 16px !important;
    font-weight: 600 !important;
    font-size: 1.0rem !important;
    letter-spacing: 0.2px !important;
    transition: all 0.18s ease !important;
    border-left: 3px solid transparent !important;
}
[data-testid="stSidebarNavLink"]:hover {
    background: rgba(229,9,20,0.1) !important;
    border-left: 3px solid rgba(229,9,20,0.45) !important;
    padding-left: 20px !important;
}
[data-testid="stSidebarNavLink"][aria-selected="true"] {
    background: rgba(229,9,20,0.18) !important;
    border-left: 4px solid #e50914 !important;
    color: #fff !important;
    padding-left: 20px !important;
}

h1, h2, h3 { font-family: 'Cinzel', serif !important; letter-spacing: 1px; }

.stButton > button {
    background: #e50914 !important; color: white !important; border: none !important;
    border-radius: 6px !important; font-weight: 600 !important; font-size: 0.8rem !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    background: #b0070f !important; transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(229,9,20,0.4) !important;
}
[data-testid="stPopover"] > button {
    background: #1c1c1c !important; color: #f5c518 !important;
    border: 1px solid #2c2c2c !important; font-size: 0.78rem !important;
}
/* Force consistent popover size regardless of screen position */
div[data-testid="stPopover"] div[role="dialog"],
div[data-testid="stPopoverContent"] {
    min-width: 260px !important;
    max-width: 280px !important;
    width: 260px !important;
}
/* Hide slider min/max end labels inside popovers */
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"] { display: none !important; }
.stTextInput input, .stNumberInput input {
    background: #141414 !important; border: 1px solid #282828 !important;
    border-radius: 8px !important; color: #f0f0f0 !important;
}
[data-baseweb="select"] > div { background: #141414 !important; border-color: #282828 !important; }
[data-baseweb="tag"] { background: #e50914 !important; }
[data-testid="metric-container"] {
    background: #141414; border-radius: 10px; padding: 10px; border: 1px solid #202020;
}
img { border-radius: 8px !important; }
hr { border-color: #181818 !important; margin: 8px 0 !important; }
/* Keep sidebar toggle always accessible */
[data-testid="collapsedControl"] { display: flex !important; visibility: visible !important; }
#MainMenu, footer { display: none; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0a0a0a; }
::-webkit-scrollbar-thumb { background: #282828; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #e50914; }

/* ── Home page clickable nav cards ── */
.home-nav-card { transition: background 0.18s ease, transform 0.18s ease, box-shadow 0.18s ease !important; }
a:hover .home-nav-card {
    background: #181818 !important;
    transform: translateY(-3px);
    box-shadow: 0 14px 40px rgba(0,0,0,0.55) !important;
}

/* ── Compact sidebar filter controls (browse page) ── */
[data-testid="stSidebar"] .stSlider,
[data-testid="stSidebar"] .stMultiSelect,
[data-testid="stSidebar"] .stSelectbox,
[data-testid="stSidebar"] .stTextInput { margin-bottom: 2px !important; }
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label {
    font-size: 0.72rem !important;
    color: #666 !important;
    margin-bottom: 0 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.4px !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] { padding-bottom: 4px !important; }
</style>
""", unsafe_allow_html=True)


# ── UI helpers ────────────────────────────────────────────────────────────────

def stars(rating: float) -> str:
    full = int(rating)
    half = 1 if (rating - full) >= 0.5 else 0
    return "★" * full + ("½" if half else "") + "☆" * (5 - full - half)

def film_strip(label: str) -> str:
    hole = '<div style="width:9px;height:15px;background:#000;border-radius:2px;border:1.5px solid #1e1e1e"></div>'
    holes = '<div style="display:flex;gap:5px">' + hole * 10 + "</div>"
    return (
        f'<div style="background:#0f0f0f;border-top:2px solid #1a1a1a;border-bottom:2px solid #1a1a1a;'
        f'padding:8px 16px;display:flex;align-items:center;justify-content:space-between;margin-bottom:20px">'
        f'{holes}'
        f'<span style="font-family:Cinzel,serif;color:#f5c518;letter-spacing:8px;font-size:0.78rem;font-weight:700">'
        f'{label}</span>'
        f'{holes}</div>'
    )

def poster_placeholder(mid: int, display_title: str, year) -> str:
    c1, c2 = POSTER_GRADIENTS[mid % len(POSTER_GRADIENTS)]
    safe = display_title.replace('"', '').replace("'", "")[:42]
    yr_html = (f'<div style="color:rgba(255,255,255,0.45);font-size:0.68rem;margin-top:6px">{int(year)}</div>'
               if year and not pd.isna(year) else "")
    return (
        f'<div style="width:100%;padding-bottom:150%;position:relative;border-radius:10px;overflow:hidden;'
        f'background:linear-gradient(160deg,{c1} 0%,{c2} 100%);box-shadow:0 6px 24px rgba(0,0,0,0.8)">'
        f'<div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;'
        f'justify-content:center;padding:12px;text-align:center">'
        f'<div style="font-size:2.2rem;line-height:1">🎬</div>'
        f'<div style="color:#fff;font-weight:700;font-size:0.74rem;margin-top:10px;line-height:1.4">{safe}</div>'
        f'{yr_html}</div></div>'
    )

def small_poster_html(mid: int) -> str:
    c1, c2 = POSTER_GRADIENTS[mid % len(POSTER_GRADIENTS)]
    return (
        f'<div style="width:70px;height:105px;border-radius:8px;overflow:hidden;flex-shrink:0;'
        f'background:linear-gradient(160deg,{c1},{c2});display:flex;align-items:center;'
        f'justify-content:center;font-size:1.8rem;box-shadow:0 4px 14px rgba(0,0,0,0.7)">🎬</div>'
    )

def poster_card_html(poster_url: str, mid: int, display_title: str, year) -> str:
    """Poster with fixed 2:3 aspect ratio so real images and placeholders stay aligned."""
    if poster_url and str(poster_url).strip() not in ("", "N/A"):
        safe_url = str(poster_url).replace('"', '%22')
        return (
            f'<div style="width:100%;padding-bottom:150%;position:relative;'
            f'border-radius:10px;overflow:hidden;box-shadow:0 6px 24px rgba(0,0,0,0.8)">'
            f'<img src="{safe_url}" style="position:absolute;top:0;left:0;'
            f'width:100%;height:100%;object-fit:cover;" loading="lazy" />'
            f'</div>'
        )
    return poster_placeholder(mid, display_title, year)


# ── OMDB integration ──────────────────────────────────────────────────────────

def _load_omdb_cache() -> dict:
    if OMDB_CACHE_PATH.exists():
        try:
            return json.loads(OMDB_CACHE_PATH.read_text())
        except Exception:
            return {}
    return {}

def _save_omdb_cache(cache: dict):
    OMDB_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    OMDB_CACHE_PATH.write_text(json.dumps(cache))

def fetch_omdb(display_title: str, year=None) -> dict:
    """Fetch from OMDB (using formatted display title), cached to disk."""
    api_key = st.session_state.get("omdb_key", "")
    if not api_key:
        return {}
    # Strip year from display title for OMDB query
    clean = re.sub(r'\s*\(\d{4}\)\s*$', '', display_title).strip()
    cache_key = f"{clean}|{year or ''}"
    if cache_key in st.session_state.omdb_cache:
        return st.session_state.omdb_cache[cache_key]
    try:
        params = {"t": clean, "apikey": api_key, "type": "movie"}
        if year and not pd.isna(year):
            params["y"] = int(year)
        resp = requests.get("http://www.omdbapi.com/", params=params, timeout=5)
        data = resp.json()
        result = {
            "poster":      data.get("Poster", ""),
            "director":    data.get("Director", ""),
            "actors":      data.get("Actors", ""),
            "genre":       data.get("Genre", ""),
            "plot":        data.get("Plot", ""),
            "runtime":     data.get("Runtime", ""),
            "imdb_rating": data.get("imdbRating", ""),
        } if data.get("Response") == "True" else {}
        st.session_state.omdb_cache[cache_key] = result
        _save_omdb_cache(st.session_state.omdb_cache)
        return result
    except Exception:
        return {}


def fetch_tmdb(tmdb_id) -> dict:
    """Fetch poster, director, cast, genres and plot from TMDB using its numeric ID."""
    api_key = st.session_state.get("tmdb_key", "")
    if not api_key or tmdb_id is None or (isinstance(tmdb_id, float) and pd.isna(tmdb_id)):
        return {}
    cache_key = f"tmdb|{int(tmdb_id)}"
    if cache_key in st.session_state.omdb_cache:
        return st.session_state.omdb_cache[cache_key]
    try:
        resp = requests.get(
            f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}",
            params={"api_key": api_key, "append_to_response": "credits"},
            timeout=5,
        )
        data = resp.json()
        if data.get("id"):
            pp = data.get("poster_path", "")
            directors = [c["name"] for c in data.get("credits", {}).get("crew", [])
                         if c.get("job") == "Director"]
            cast = [c["name"] for c in data.get("credits", {}).get("cast", [])[:5]]
            result = {
                "poster":      f"https://image.tmdb.org/t/p/w300{pp}" if pp else "",
                "director":    ", ".join(directors),
                "actors":      ", ".join(cast),
                "genre":       ", ".join(g["name"] for g in data.get("genres", [])),
                "plot":        data.get("overview", ""),
                "runtime":     f"{data['runtime']} min" if data.get("runtime") else "",
                "imdb_rating": str(round(data.get("vote_average", 0), 1)),
            }
        else:
            result = {}
        st.session_state.omdb_cache[cache_key] = result
        _save_omdb_cache(st.session_state.omdb_cache)
        return result
    except Exception:
        return {}

def fetch_movie_meta(display_title: str, year=None, tmdb_id=None) -> dict:
    """TMDB first (exact ID, high quality), OMDB as fallback."""
    if tmdb_id is not None and not (isinstance(tmdb_id, float) and pd.isna(tmdb_id)):
        result = fetch_tmdb(tmdb_id)
        if result.get("poster"):
            return result
    return fetch_omdb(display_title, year)

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_catalog() -> pd.DataFrame:
    df = pd.read_csv(EMBEDDINGS_PATH, usecols=["movieId", "title"]).drop_duplicates("movieId")

    # Apply display title formatting
    df["display_title"] = df["title"].apply(format_display_title)

    # Extract year from original title (format is always "Title (YEAR)")
    df["year"] = df["title"].str.extract(r'\((\d{4})\)\s*$').astype(float)

    # Decade column (always available once year is extracted)
    df["decade"] = (df["year"] // 10 * 10).astype("Int64")

    # Sort key for alphabetical ordering (ignores leading articles)
    df["sort_key"] = df["display_title"].apply(sort_key)

    # Merge genres from processed data if available
    if MOVIES_PATH.exists():
        proc = pd.read_csv(MOVIES_PATH)[["movieId", "genres"]]
        df = df.merge(proc, on="movieId", how="left")
    else:
        df["genres"] = ""

    # Merge TMDB/IMDB IDs from links.csv (downloaded with MovieLens data)
    if LINKS_PATH.exists():
        links = pd.read_csv(LINKS_PATH)[["movieId", "tmdbId"]].dropna(subset=["tmdbId"])
        links["tmdbId"] = links["tmdbId"].astype(int)
        df = df.merge(links, on="movieId", how="left")
    else:
        df["tmdbId"] = np.nan

    # Merge rating stats if available
    if STATS_PATH.exists():
        stats = pd.read_csv(STATS_PATH)[["movieId", "mean_rating", "count"]]
        df = df.merge(stats, on="movieId", how="left")
    else:
        df["mean_rating"] = np.nan
        df["count"] = np.nan
    
    # Merge poster mapping if available
    if POSTER_MAPPING_PATH.exists():
        posters = pd.read_csv(POSTER_MAPPING_PATH)[["movieId", "poster_url"]].drop_duplicates("movieId")
        df = df.merge(posters, on="movieId", how="left")
    else:
        df["poster_url"] = ""

    return df.reset_index(drop=True)

@st.cache_resource
def load_embeddings():
    if not EMBEDDINGS_PATH.exists():
        return None
    return load_embedding_table(str(EMBEDDINGS_PATH))

def get_recommendations(top_k: int, min_rating: float) -> pd.DataFrame | None:
    result = load_embeddings()
    if result is None:
        return None
    _, emb_matrix, movieid_to_idx, movie_ids, titles = result
    user_df = pd.DataFrame([
        {"movieId": mid, "rating": info["rating"]}
        for mid, info in st.session_state.ratings.items()
    ])
    try:
        query_vec, _ = build_user_query_vector(
            user_ratings=user_df, emb_matrix=emb_matrix,
            movieid_to_idx=movieid_to_idx, min_rating=min_rating, weighted=True,
        )
    except ValueError:
        return None
    sims = emb_matrix @ query_vec
    rated_mask = np.array([mid in st.session_state.ratings for mid in movie_ids])
    sims[rated_mask] = -np.inf
    top_idx = np.argsort(-sims)[:top_k]
    return pd.DataFrame({
        "movieId": movie_ids[top_idx],
        "title":   titles[top_idx],
        "score":   sims[top_idx],
    }).reset_index(drop=True)


@st.cache_data
def count_eligible_users(
    ratings_path_str: str,
    leave_k: int,
    min_train_ratings: int,
    min_relevant_test: int,
    relevance_threshold: float,
) -> tuple[int, int, int]:
    """Returns (eligible_users, users_with_enough_history, total_users)."""
    rp = Path(ratings_path_str)
    if not rp.exists():
        return 0, 0, 0
    ratings = pd.read_csv(rp)
    if not {"userId", "movieId", "rating", "timestamp"}.issubset(ratings.columns):
        return 0, 0, 0
    ratings = ratings.sort_values(["userId", "timestamp"])
    total = 0
    enough_history = 0
    eligible = 0
    for _, user_hist in ratings.groupby("userId", sort=False):
        total += 1
        if len(user_hist) < (leave_k + min_train_ratings):
            continue
        enough_history += 1
        test = user_hist.iloc[-leave_k:]
        if (test["rating"] >= relevance_threshold).sum() >= min_relevant_test:
            eligible += 1
    return eligible, enough_history, total


@st.cache_data(show_spinner=False)
def run_sensitivity_analysis(
    ratings_path_str: str,
    embeddings_path_str: str,
    vary_param: str,
    base_top_k: int,
    base_leave_k: int,
    base_min_train: int,
    base_min_rel: int,
    base_threshold: float,
    base_max_windows: int,
) -> pd.DataFrame:
    """Run evaluation across a range of one parameter, holding others fixed.

    For leave_k sweeps, min_train_ratings and min_relevant_test both scale with K so
    that the user-count curve is driven purely by the history requirement, not by the
    easier-to-satisfy relevance filter at larger K.
    """
    _param_ranges: dict[str, list] = {
        "top_k":               list(range(5, 51)),           # 5–50, step 1
        "leave_k":             list(range(1, 21)),           # 1–20, step 1
        "min_relevant_test":   list(range(1, min(base_leave_k + 1, 11))),  # step 1
        "relevance_threshold": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    }
    vals = _param_ranges.get(vary_param, [])
    rows = []
    from evaluation.evaluate import evaluate_temporal_leave_k_out
    for val in vals:
        kwargs: dict = dict(
            ratings_path=ratings_path_str,
            embeddings_path=embeddings_path_str,
            top_k=base_top_k,
            leave_k=base_leave_k,
            min_train_ratings=base_min_train,
            min_relevant_test=base_min_rel,
            relevance_threshold=base_threshold,
            max_windows=base_max_windows,
        )
        kwargs[vary_param] = val
        if vary_param == "leave_k":
            # Scale min_train and min_rel proportionally with K so the user-count
            # curve reflects the stricter history requirement, not the easier
            # min_relevant_test filter (which grows easier to satisfy as K increases).
            kwargs["min_train_ratings"] = max(10, int(val) + 10)
            prop_min_rel = max(1, round(base_min_rel / base_leave_k * int(val)))
            kwargs["min_relevant_test"] = min(prop_min_rel, int(val))
        try:
            s, _ = evaluate_temporal_leave_k_out(**kwargs)
            rows.append({
                vary_param:                 val,
                "Precision@K":              s.get("precision_at_k", 0),
                "NDCG@K (Binary)":          s.get("ndcg_at_k", 0),
                "Graded NDCG@K":            s.get("graded_ndcg_at_k", 0),
                "Hit Rate@K":               s.get("hit_rate_at_k", 0),
                "Pairwise Rank Acc":        s.get("pairwise_rank_acc_at_k", 0),
                "Dislike Rate@K":           s.get("dislike_rate_at_k", 0),
                "Users Evaluated":          s.get("users_evaluated", 0),
            })
        except Exception:
            pass
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index(vary_param)


# ── Session state init ────────────────────────────────────────────────────────

_defaults = {
    "ratings":          {},   # {movieId: {display_title, rating, genres, year, director, tmdbId}}
    "recs":             None,
    "browse_page":      0,
    "last_filter_key":  None,
    "tmdb_key":         "",
    "omdb_key":         "",
    "omdb_cache":       _load_omdb_cache(),
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Load catalog (module-level, shared across pages) ──────────────────────────

catalog    = load_catalog()
n_rated    = len(st.session_state.ratings)
n_liked    = sum(1 for v in st.session_state.ratings.values() if v["rating"] >= 4.0)
has_genres = catalog["genres"].notna().any() and catalog["genres"].ne("").any()
has_stats  = not catalog["mean_rating"].isna().all()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME (Landing)
# ═══════════════════════════════════════════════════════════════════════════════

def page_home():
    # Hero
    st.markdown(
        '<div style="text-align:center;padding:56px 0 36px">'
        '<div style="font-size:4.5rem;line-height:1;margin-bottom:14px">🎬</div>'
        '<div style="font-family:Cinzel,serif;font-size:3.2rem;font-weight:900;'
        'color:#f0f0f0;letter-spacing:8px;line-height:1">CINE'
        '<span style="color:#e50914">MATCH</span></div>'
        '<div style="font-family:Cinzel,serif;color:#f5c518;letter-spacing:10px;'
        'font-size:0.72rem;margin:10px 0 28px;text-transform:uppercase">Film Discovery Engine</div>'
        '<div style="max-width:580px;margin:0 auto">'
        '<p style="color:#666;font-size:1.05rem;line-height:1.85;margin:0">'
        'Your AI-powered cinema companion. Rate films you\'ve seen and CineMatch '
        'learns your taste to surface hidden gems you\'ll love — no sign-up required.'
        '</p></div></div>',
        unsafe_allow_html=True,
    )

    # Feature cards — the entire block is the clickable link
    _features = [
        ("🎬", "Browse",   "Explore 9,000+ films. Filter by genre, decade, year, and rating.", "#e50914", "/browse"),
        ("⭐", "Rate",     "Rate movies you've watched to build your personal taste profile.",  "#f5c518", "/my-ratings"),
        ("🎯", "For You",  "Get AI-curated picks based on your ratings — powered by embeddings.", "#1a6eb5", "/for-you"),
        ("📊", "Evaluate", "See precision, recall, NDCG and more from our rigorous evaluation.", "#1a7a40", "/evaluation"),
    ]
    cols = st.columns(4, gap="medium")
    for col, (icon, title, desc, color, url) in zip(cols, _features):
        with col:
            st.markdown(
                f'<a href="{url}" target="_self" style="text-decoration:none;display:block">'
                f'<div class="home-nav-card" style="background:#111;border:1px solid #1e1e1e;'
                f'border-top:3px solid {color};border-radius:12px;padding:26px 18px;'
                f'text-align:center;min-height:185px;cursor:pointer">'
                f'<div style="font-size:2rem;margin-bottom:12px">{icon}</div>'
                f'<div style="font-family:Cinzel,serif;color:#f0f0f0;font-size:0.88rem;'
                f'font-weight:700;letter-spacing:2px;margin-bottom:10px">{title}</div>'
                f'<p style="color:#555;font-size:0.78rem;line-height:1.55;margin:0">{desc}</p>'
                f'</div></a>',
                unsafe_allow_html=True,
            )

    # How it works strip
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(film_strip("HOW IT WORKS"), unsafe_allow_html=True)

    _steps = [
        ("01", "Browse & Rate",        "Open Browse. Find films you've seen and rate them 0.5–5 stars."),
        ("02", "Build Your Profile",   "Rate 5+ films at 4 ★ or higher — CineMatch learns what you love."),
        ("03", "Get Recommendations",  "Head to For You and hit Get My Recommendations for your AI-curated list."),
        ("04", "Explore the Science",  "Curious how accurate it is? Check Evaluation for precision, recall & more."),
    ]
    cols2 = st.columns(4, gap="medium")
    for col, (num, title, desc) in zip(cols2, _steps):
        with col:
            st.markdown(
                f'<div style="background:#0f0f0f;border:1px solid #181818;border-radius:10px;'
                f'padding:22px 16px;min-height:155px">'
                f'<div style="font-family:Cinzel,serif;font-size:2.2rem;font-weight:900;'
                f'color:#e50914;margin-bottom:10px;line-height:1">{num}</div>'
                f'<div style="font-weight:600;color:#f0f0f0;font-size:0.88rem;margin-bottom:8px">{title}</div>'
                f'<p style="color:#4a4a4a;font-size:0.77rem;line-height:1.55;margin:0">{desc}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Navigation hint
    st.markdown(
        '<div style="text-align:center;padding:40px 0 24px">'
        '<div style="background:#111;border:1px solid #1e1e1e;border-radius:14px;'
        'padding:28px 32px;max-width:520px;margin:0 auto">'
        '<div style="font-family:Cinzel,serif;color:#f5c518;letter-spacing:4px;'
        'font-size:0.7rem;margin-bottom:14px;text-transform:uppercase">Getting Started</div>'
        '<p style="color:#777;font-size:0.88rem;line-height:1.8;margin:0">'
        'Use the <b style="color:#f0f0f0">sidebar on the left ←</b> to switch between sections.<br>'
        'New here? Start with <b style="color:#e50914">Browse</b> to find and rate your first films.'
        '</p></div></div>',
        unsafe_allow_html=True,
    )

    # Catalog stats footer
    st.markdown(
        f'<div style="text-align:center;padding:8px 0 40px">'
        f'<span style="color:#252525;font-size:0.78rem">'
        f'{len(catalog):,} films in catalog &nbsp;·&nbsp; '
        f'Semantic embeddings via sentence-transformers &nbsp;·&nbsp; '
        f'Contrastive fine-tuning on real user ratings'
        f'</span></div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: BROWSE
# ═══════════════════════════════════════════════════════════════════════════════

def page_browse():
    # ── Sidebar filters ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("**Filters**")

        search = st.text_input("🔍 Title", placeholder="Search titles…",
                               label_visibility="collapsed")

        # Decade (always available)
        decade_options = sorted(catalog["decade"].dropna().unique().astype(int).tolist())
        decade_labels  = {d: f"{d}s" for d in decade_options}
        selected_decades = st.multiselect(
            "📅 Decade", decade_options,
            format_func=lambda d: decade_labels[d],
        )

        # Year range (fine-grained, always available)
        valid_years = catalog["year"].dropna()
        if not valid_years.empty:
            yr_min, yr_max = int(valid_years.min()), int(valid_years.max())
            year_range = st.slider("📆 Year range", yr_min, yr_max, (yr_min, yr_max))
        else:
            year_range = None

        # Genre filter (populated from processed data; empty until pipeline has run)
        all_genres = sorted({
            g.strip()
            for gs in catalog["genres"].dropna()
            for g in str(gs).replace("|", ",").split(",")
            if g.strip() and g.strip().lower() not in ("(no genres listed)", "unknown", "")
        })
        selected_genres: list[str] = st.multiselect("🎭 Genre", all_genres)

        # Avg rating floor (if stats available)
        min_avg_rating = 0.0
        if has_stats:
            min_avg_rating = st.slider("⭐ Min avg rating", 0.0, 5.0, 0.0, 0.5)

        # Director (only when OMDB key is active)
        director_filter = ""
        if st.session_state.omdb_key:
            director_filter = st.text_input("🎬 Director", placeholder="e.g. Spielberg…")

        # Sort
        sort_opts = ["Title A→Z", "Title Z→A", "Year (Newest)", "Year (Oldest)"]
        if has_stats:
            sort_opts += ["Highest Rated", "Lowest Rated", "Most Ratings"]
        sort_by = st.selectbox("Sort by", sort_opts)

    # ── Apply filters ─────────────────────────────────────────────────────────
    flt = catalog.copy()

    if search.strip():
        flt = flt[flt["display_title"].str.contains(
            search.strip(), case=False, na=False, regex=False)]

    if selected_decades:
        flt = flt[flt["decade"].isin(selected_decades)]

    if year_range:
        lo, hi = year_range
        flt = flt[flt["year"].isna() | flt["year"].between(lo, hi)]

    if selected_genres:
        flt = flt[flt["genres"].apply(
            lambda g: any(s.lower() in str(g).lower() for s in selected_genres)
        )]

    if min_avg_rating > 0 and has_stats:
        flt = flt[flt["mean_rating"].isna() | (flt["mean_rating"] >= min_avg_rating)]

    # Sort
    sort_map = {
        "Title A→Z":      ("sort_key",    True),
        "Title Z→A":      ("sort_key",    False),
        "Year (Newest)":  ("year",        False),
        "Year (Oldest)":  ("year",        True),
        "Highest Rated":  ("mean_rating", False),
        "Lowest Rated":   ("mean_rating", True),
        "Most Ratings":   ("count",       False),
    }
    if sort_by in sort_map:
        col_s, asc_s = sort_map[sort_by]
        flt = flt.sort_values(col_s, ascending=asc_s, na_position="last")

    # Stable-sort: movies with a local poster bubble to the front within each page
    _has_poster = flt["poster_url"].notna() & flt["poster_url"].ne("")
    flt = flt.assign(_has_poster=_has_poster).sort_values(
        "_has_poster", ascending=False, kind="stable"
    ).drop(columns=["_has_poster"])

    # Reset pagination when filters change
    fk = (search, tuple(selected_decades), year_range, tuple(selected_genres),
          min_avg_rating, director_filter, sort_by)
    if st.session_state.last_filter_key != fk:
        st.session_state.browse_page = 0
        st.session_state.last_filter_key = fk

    # ── Render ────────────────────────────────────────────────────────────────
    st.markdown(film_strip("NOW SHOWING"), unsafe_allow_html=True)
    st.markdown(
        '<h1 style="font-size:1.9rem;margin:0 0 4px;color:#f0f0f0">Browse Movies</h1>'
        f'<p style="color:#444;font-size:0.83rem;margin-bottom:20px">'
        f'{len(catalog):,} titles in catalog</p>',
        unsafe_allow_html=True,
    )

    total       = len(flt)
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    cur         = min(st.session_state.browse_page, total_pages - 1)
    page_df     = flt.iloc[cur * PAGE_SIZE : (cur + 1) * PAGE_SIZE]

    if total == 0:
        st.info("No movies match your filters.")
        return

    rows = [list(page_df.iloc[i:i+4].iterrows()) for i in range(0, len(page_df), 4)]
    for row_data in rows:
        cols = st.columns(4, gap="medium")
        for col, (_, mv) in zip(cols, row_data):
            mid   = int(mv["movieId"])
            dtitle = str(mv["display_title"])
            year  = mv.get("year")
            gs    = str(mv.get("genres", "") or "")
            avg_r = mv.get("mean_rating")
            cnt   = mv.get("count")

            
            tmdb_id = mv.get("tmdbId")
            omdb = fetch_movie_meta(dtitle, year, tmdb_id)

            # Director filter (OMDB/TMDB-gated)
            if director_filter.strip() and omdb:
                if director_filter.strip().lower() not in (omdb.get("director","") or "").lower():
                    continue

            already = mid in st.session_state.ratings
            cur_r   = st.session_state.ratings[mid]["rating"] if already else None

            with col:
                local_poster_url = mv.get("poster_url", "")
                poster_url = (
                    local_poster_url
                    if pd.notna(local_poster_url) and str(local_poster_url).strip()
                    else (omdb or {}).get("poster", "")
                )
                st.markdown(poster_card_html(poster_url, mid, dtitle, year),
                            unsafe_allow_html=True)

                # Title
                st.markdown(
                    f'<div style="font-size:0.8rem;font-weight:600;color:#f0f0f0;'
                    f'margin-top:6px;line-height:1.35;height:2.8em;overflow:hidden">{dtitle}</div>',
                    unsafe_allow_html=True,
                )

                # Meta
                meta = []
                if year and not pd.isna(year): meta.append(str(int(year)))
                if avg_r and not pd.isna(avg_r): meta.append(f"⭐ {avg_r:.1f}")
                if cnt and not pd.isna(cnt): meta.append(f"{int(cnt):,} ratings")
                if omdb and omdb.get("runtime"): meta.append(omdb["runtime"])
                if meta:
                    st.markdown(
                        f'<div style="color:#444;font-size:0.68rem;margin-bottom:4px">'
                        f'{" · ".join(meta)}</div>',
                        unsafe_allow_html=True,
                    )

                # Rate popover
                pop_lbl = f"★ {cur_r}" if already else "+ Rate"
                with st.popover(pop_lbl, use_container_width=True):
                    st.markdown(f"**{dtitle}**")
                    if omdb:
                        if omdb.get("director"): st.caption(f"Dir. {omdb['director']}")
                        if omdb.get("actors"):   st.caption(f"Cast: {omdb['actors'][:80]}")
                        if omdb.get("plot") and omdb["plot"] not in ("", "N/A"):
                            st.caption(omdb["plot"][:180])
                        if omdb.get("imdb_rating") and omdb["imdb_rating"] not in ("", "N/A"):
                            st.caption(f"IMDb: {omdb['imdb_rating']}/10")

                    chosen = st.slider(
                        "Your rating",
                        min_value=0.5, max_value=5.0, step=0.5,
                        value=cur_r if cur_r is not None else 3.5,
                        key=f"sl_{mid}",
                    )
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("Save", key=f"sv_{mid}", type="primary",
                                     use_container_width=True):
                            st.session_state.ratings[mid] = {
                                "display_title": dtitle,
                                "rating":   float(chosen),
                                "genres":   gs,
                                "year":     None if (year is None or pd.isna(year)) else int(year),
                                "director": (omdb or {}).get("director", ""),
                                "tmdbId":   None if (tmdb_id is None or (isinstance(tmdb_id, float) and pd.isna(tmdb_id))) else int(tmdb_id),
                            }
                            st.session_state.recs = None
                            st.rerun()
                    with c2:
                        if already and st.button("Remove", key=f"rm_{mid}",
                                                 use_container_width=True):
                            del st.session_state.ratings[mid]
                            st.session_state.recs = None
                            st.rerun()

    # Pagination
    st.markdown("---")
    pc1, pc2, pc3 = st.columns([2, 3, 2])
    with pc1:
        if st.button("← Previous", disabled=(cur == 0), use_container_width=True):
            st.session_state.browse_page = cur - 1
            st.rerun()
    with pc2:
        st.markdown(
            f'<div style="text-align:center;color:#444;font-size:0.82rem;padding-top:8px">'
            f'Page {cur+1} of {total_pages} &nbsp;·&nbsp; {total:,} titles</div>',
            unsafe_allow_html=True,
        )
    with pc3:
        if st.button("Next →", disabled=(cur >= total_pages-1), use_container_width=True):
            st.session_state.browse_page = cur + 1
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MY RATINGS
# ═══════════════════════════════════════════════════════════════════════════════

def page_my_ratings():
    # ── Sidebar filters ───────────────────────────────────────────────────────
    with st.sidebar:
        if n_rated > 0:
            st.markdown("**Sort & Filter**")
            my_sort = st.selectbox(
                "Sort by",
                ["My Rating ↓", "My Rating ↑", "Title A→Z", "Year (Newest)", "Year (Oldest)"],
            )
            # Genre filter from rated movies
            genre_pool = sorted({
                g.strip()
                for info in st.session_state.ratings.values()
                for g in str(info.get("genres","")).replace("|",",").split(",")
                if g.strip() and g.strip().lower() not in ("","unknown","(no genres listed)")
            })
            my_genres = st.multiselect("🎭 Genre", genre_pool) if genre_pool else []

            # Director filter from rated movies
            dir_pool = sorted({
                info.get("director","").strip()
                for info in st.session_state.ratings.values()
                if info.get("director","").strip() not in ("","N/A")
            })
            my_director = st.selectbox("🎬 Director", ["All"] + dir_pool) if dir_pool else "All"
        else:
            my_sort, my_genres, my_director = "My Rating ↓", [], "All"

    # ── Render ────────────────────────────────────────────────────────────────
    st.markdown(film_strip("YOUR COLLECTION"), unsafe_allow_html=True)
    st.markdown(
        '<h1 style="font-size:1.9rem;margin:0 0 20px;color:#f0f0f0">My Ratings</h1>',
        unsafe_allow_html=True,
    )

    if not st.session_state.ratings:
        st.markdown(
            '<div style="text-align:center;padding:80px 0;color:#333">'
            '<div style="font-size:5rem">🎥</div>'
            '<p style="font-size:1.1rem;margin-top:20px;color:#444">No movies rated yet.</p>'
            '<p style="color:#333">Head to <b>Browse</b> to start your collection.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # Stats bar
    avg_u = sum(v["rating"] for v in st.session_state.ratings.values()) / n_rated
    unique_genres = len({
        g.strip()
        for info in st.session_state.ratings.values()
        for g in str(info.get("genres","")).replace("|",",").split(",") if g.strip()
    })
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rated", n_rated)
    c2.metric("Loved (4★+)", n_liked)
    c3.metric("Your Avg", f"{avg_u:.1f} ★")
    c4.metric("Genres", unique_genres)
    st.markdown("---")

    # Build, filter, sort
    items = [{"mid": mid, **info} for mid, info in st.session_state.ratings.items()]

    if my_genres:
        items = [it for it in items if any(
            g.lower() in str(it.get("genres","")).lower() for g in my_genres
        )]
    if my_director != "All":
        items = [it for it in items if it.get("director","") == my_director]

    sort_fn = {
        "My Rating ↓":   lambda x: -x["rating"],
        "My Rating ↑":   lambda x:  x["rating"],
        "Title A→Z":     lambda x:  sort_key(x.get("display_title", x.get("title",""))),
        "Year (Newest)": lambda x: -(x.get("year") or 0),
        "Year (Oldest)": lambda x:  (x.get("year") or 9999),
    }
    items.sort(key=sort_fn.get(my_sort, lambda x: -x["rating"]))

    for item in items:
        mid      = item["mid"]
        dtitle   = item.get("display_title", item.get("title", ""))
        rating   = item["rating"]
        gs       = str(item.get("genres","") or "")
        year     = item.get("year")
        director = item.get("director","")

        tmdb_id = item.get("tmdbId")
        omdb = fetch_movie_meta(dtitle, year, tmdb_id)
        catalog_row = catalog[catalog["movieId"] == mid]
        local_poster_url = (
            catalog_row["poster_url"].iloc[0]
            if not catalog_row.empty and "poster_url" in catalog_row.columns
            else ""
        )
        
        poster_url = (
            local_poster_url
            if pd.notna(local_poster_url) and str(local_poster_url).strip()
            else (omdb or {}).get("poster", "")
        )

        col_img, col_info, col_r, col_act = st.columns([1, 5, 2, 1])

        with col_img:
            if poster_url and poster_url not in ("","N/A"):
                st.image(poster_url, width=70)
            else:
                st.markdown(small_poster_html(mid), unsafe_allow_html=True)

        with col_info:
            st.markdown(
                f'<div style="font-weight:600;font-size:0.95rem;color:#f0f0f0">{dtitle}</div>',
                unsafe_allow_html=True,
            )
            meta = []
            if year: meta.append(str(int(year)))
            d = director or (omdb or {}).get("director","")
            if d and d not in ("","N/A"): meta.append(f"Dir. {d}")
            if gs: meta.append(gs.replace("|"," · ")[:60])
            if meta:
                st.markdown(
                    f'<div style="color:#444;font-size:0.78rem;margin-top:3px">{" · ".join(meta)}</div>',
                    unsafe_allow_html=True,
                )

        with col_r:
            st.markdown(
                f'<div style="color:#f5c518;font-size:1rem;padding-top:6px">{stars(rating)} '
                f'<span style="color:#444;font-size:0.8rem">({rating})</span></div>',
                unsafe_allow_html=True,
            )
            with st.popover("Edit", use_container_width=True):
                new_r = st.slider(
                    "Update rating",
                    min_value=0.5, max_value=5.0, step=0.5,
                    value=rating,
                    key=f"edit_sl_{mid}",
                )
                if st.button("Update", key=f"upd_{mid}", type="primary"):
                    st.session_state.ratings[mid]["rating"] = float(new_r)
                    st.session_state.recs = None
                    st.rerun()

        with col_act:
            if st.button("✕", key=f"del_{mid}", help="Remove"):
                del st.session_state.ratings[mid]
                st.session_state.recs = None
                st.rerun()

        st.markdown("<hr style='margin:5px 0;border-color:#141414'>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: FOR YOU
# ═══════════════════════════════════════════════════════════════════════════════

def page_for_you():
    # ── Sidebar prefs ─────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("**Preferences**")
        for_k = st.slider("# of recommendations", 5, 50, 10, 5)
        for_min = st.select_slider(
            "Min ★ to learn from",
            options=[3.0, 3.5, 4.0, 4.5, 5.0], value=4.0,
        )

    # ── Render ────────────────────────────────────────────────────────────────
    st.markdown(film_strip("CURATED FOR YOU"), unsafe_allow_html=True)
    st.markdown(
        '<h1 style="font-size:1.9rem;margin:0 0 4px;color:#f0f0f0">For You</h1>',
        unsafe_allow_html=True,
    )

    if not EMBEDDINGS_PATH.exists():
        st.warning("Embeddings not built. Run `embeddings/build_embeddings.py` first.")
        return

    if n_liked == 0:
        st.markdown(
            '<div style="text-align:center;padding:80px 0">'
            '<div style="font-size:5rem">⭐</div>'
            '<p style="color:#444;font-size:1.1rem;margin-top:20px">'
            'Rate at least one movie <b>4★ or higher</b> to unlock recommendations.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # Taste profile pills
    genre_counts: dict[str, int] = {}
    for info in st.session_state.ratings.values():
        if info["rating"] >= for_min:
            for g in str(info.get("genres","")).replace("|",",").split(","):
                g = g.strip()
                if g:
                    genre_counts[g] = genre_counts.get(g, 0) + 1

    if genre_counts:
        top_genres = sorted(genre_counts.items(), key=lambda x: -x[1])[:6]
        pills = "".join(
            f'<span style="background:#111;border:1px solid #e50914;color:#ddd;'
            f'padding:4px 14px;border-radius:20px;font-size:0.78rem;margin:3px;'
            f'display:inline-block">{g} <b style="color:#e50914">{c}</b></span>'
            for g, c in top_genres
        )
        st.markdown(
            '<p style="color:#444;font-size:0.82rem;margin-bottom:6px">'
            'Taste profile — genres from your liked films:</p>'
            f'<div style="margin-bottom:22px">{pills}</div>',
            unsafe_allow_html=True,
        )

    # ── Model quality metrics (from evaluation/evaluate.py via run.py) ──────────
    if EVAL_RESULTS_PATH.exists():
        try:
            ev = json.loads(EVAL_RESULTS_PATH.read_text())
            with st.expander("📊 Model Quality Metrics", expanded=False):
                st.markdown(
                    '<p style="color:#666;font-size:0.78rem;margin-bottom:10px">'
                    'Temporal leave-k-out evaluation — how well the model generalises '
                    f'across {ev.get("users_evaluated", "?")} users.</p>',
                    unsafe_allow_html=True,
                )
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Precision@K",      f'{ev.get("precision_at_k", 0):.3f}')
                mc2.metric("Graded NDCG@K",    f'{ev.get("graded_ndcg_at_k", 0):.3f}')
                mc3.metric("Hit Rate@K",       f'{ev.get("hit_rate_at_k", 0):.3f}')
                mc4.metric("Pair Rank Acc",    f'{ev.get("pairwise_rank_acc_at_k", 0):.3f}')
        except Exception:
            pass

    if st.button("🎯  Get My Recommendations", type="primary"):
        with st.spinner("Analyzing your taste profile…"):
            st.session_state.recs = get_recommendations(for_k, for_min)

    recs = st.session_state.recs
    if recs is None:
        return

    if recs.empty:
        st.warning("No results — try rating more movies or lowering the taste threshold.")
        return

    st.markdown(
        f'<p style="color:#444;font-size:0.85rem;margin:16px 0 20px">'
        f'Top {len(recs)} picks based on your {n_liked} liked '
        f'film{"s" if n_liked != 1 else ""}.</p>',
        unsafe_allow_html=True,
    )

    for rank, (_, row) in enumerate(recs.iterrows(), 1):
        mid    = int(row["movieId"])
        raw_t  = str(row["title"])
        dtitle = format_display_title(raw_t)
        score  = float(row["score"])
        pct    = max(0.0, min(100.0, score * 100))
        catalog_row = catalog[catalog["movieId"] == mid]
        tmdb_id = catalog_row["tmdbId"].iloc[0] if not catalog_row.empty else None
        omdb = fetch_movie_meta(dtitle, None, tmdb_id)
        
        local_poster_url = (
            catalog_row["poster_url"].iloc[0]
            if not catalog_row.empty and "poster_url" in catalog_row.columns
            else ""
        )
        poster_url = (
            local_poster_url
            if pd.notna(local_poster_url) and str(local_poster_url).strip()
            else (omdb or {}).get("poster", "")
        )

        col_img, col_info, col_score = st.columns([1, 5, 2])

        with col_img:
            if poster_url and poster_url not in ("","N/A"):
                st.image(poster_url, width=70)
            else:
                st.markdown(small_poster_html(mid), unsafe_allow_html=True)

        with col_info:
            director = (omdb or {}).get("director","")
            genre    = (omdb or {}).get("genre","")
            plot     = (omdb or {}).get("plot","")
            st.markdown(
                f'<div style="font-family:Cinzel,serif;font-size:0.68rem;color:#e50914;'
                f'font-weight:700;letter-spacing:2px">#{rank}</div>'
                f'<div style="font-weight:600;font-size:1rem;color:#f0f0f0">{dtitle}</div>',
                unsafe_allow_html=True,
            )
            meta = []
            if director and director not in ("","N/A"): meta.append(f"Dir. {director}")
            if genre: meta.append(genre[:60])
            if meta:
                st.markdown(
                    f'<div style="color:#444;font-size:0.78rem;margin-top:2px">'
                    f'{" · ".join(meta)}</div>',
                    unsafe_allow_html=True,
                )
            if plot and plot not in ("","N/A"):
                st.markdown(
                    f'<div style="color:#555;font-size:0.75rem;margin-top:4px;line-height:1.5">'
                    f'{plot[:220]}…</div>',
                    unsafe_allow_html=True,
                )

        with col_score:
            st.markdown(
                f'<div style="text-align:right;padding-top:4px">'
                f'<div style="color:#f5c518;font-size:1.2rem;font-weight:700">{pct:.1f}%</div>'
                f'<div style="color:#333;font-size:0.7rem;margin-bottom:8px">match</div>'
                f'<div style="background:#181818;border-radius:4px;height:5px">'
                f'<div style="background:linear-gradient(90deg,#e50914,#ff8080);'
                f'border-radius:4px;height:5px;width:{int(pct)}%"></div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            already = mid in st.session_state.ratings
            with st.popover("+ Rate", use_container_width=True):
                chosen = st.slider(
                    "Rate this",
                    min_value=0.5, max_value=5.0, step=0.5,
                    value=st.session_state.ratings[mid]["rating"] if already else 3.5,
                    key=f"rec_sl_{mid}",
                )
                if st.button("Save", key=f"rec_sv_{mid}", type="primary"):
                    st.session_state.ratings[mid] = {
                        "display_title": dtitle,
                        "rating":   float(chosen),
                        "genres":   (omdb or {}).get("genre",""),
                        "year":     None,
                        "director": (omdb or {}).get("director",""),
                    }
                    st.session_state.recs = None
                    st.rerun()

        st.markdown("<hr style='margin:10px 0;border-color:#141414'>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def page_evaluation():
    # Load stored results first so sidebar sliders can default to the params used last time.
    if EVAL_RESULTS_PATH.exists():
        try:
            ev = json.loads(EVAL_RESULTS_PATH.read_text())
        except Exception:
            ev = {}
    else:
        ev = {}

    with st.sidebar:
        st.markdown("**Evaluation Parameters**")
        eval_top_k = st.slider(
            "N — recommendations shown",
            min_value=5, max_value=100,
            value=int(ev.get("top_k", 50)), step=5,
            help="How many top recommendations to score against the held-out set.",
        )
        eval_leave_k = st.slider(
            "K — movies held out per user",
            min_value=1, max_value=20,
            value=int(ev.get("leave_k", 5)), step=1,
            help="Number of each user's most recent ratings hidden as the test set.",
        )
        eval_min_train = max(10, eval_leave_k + 10)
        st.caption(f"Min training ratings: **{eval_min_train}** (auto: max(10, K+10))")
        # max(10, K+5) always exceeds K for any K ≤ 20, so this is capped to K —
        # meaning all held-out movies must be liked for a user to be included.
        eval_min_rel = 1
        st.caption(f"Min liked in held-out: **{eval_min_rel}** (users qualify if they liked ≥ 1 of their {eval_leave_k} held-out movies)")
        eval_relevance_threshold = st.select_slider(
            "Liked = rating ≥",
            options=[2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            value=float(ev.get("relevance_threshold", 3.5)),
            help="Minimum rating for a movie to count as 'liked' in the held-out test set.",
        )
        eval_max_windows = st.slider(
            "Max windows per user",
            min_value=1, max_value=10,
            value=int(ev.get("max_windows", 3)), step=1,
            help="How many sliding K-movie windows to evaluate per user. "
                 "More windows = more signal per user but longer runtime.",
        )
        st.markdown("---")
        if st.button("Re-run Evaluation", use_container_width=True,
                     help="Re-runs the full temporal leave-k-out evaluation. May take a minute."):
            with st.spinner("Running evaluation…"):
                try:
                    from evaluation.evaluate import evaluate_temporal_leave_k_out
                    summary, _ = evaluate_temporal_leave_k_out(
                        ratings_path=str(PROJECT_ROOT / "data" / "processed" / "ratings_clean.csv"),
                        embeddings_path=str(EMBEDDINGS_PATH),
                        top_k=eval_top_k,
                        leave_k=eval_leave_k,
                        min_train_ratings=eval_min_train,
                        min_relevant_test=eval_min_rel,
                        relevance_threshold=eval_relevance_threshold,
                        max_windows=eval_max_windows,
                    )
                    EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
                    EVAL_RESULTS_PATH.write_text(json.dumps(summary, indent=2))
                    st.success("Done!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")

    st.markdown(film_strip("THE NUMBERS"), unsafe_allow_html=True)
    st.markdown(
        '<h1 style="font-size:1.9rem;margin:0 0 4px;color:#f0f0f0">Evaluation Metrics</h1>'
        '<p style="color:#444;font-size:0.85rem;margin-bottom:24px">'
        'How well does our recommendation model actually work?</p>',
        unsafe_allow_html=True,
    )

    # ── Stale-params warning ───────────────────────────────────────────────────
    params_changed = ev and (
        eval_top_k != int(ev.get("top_k", eval_top_k))
        or eval_leave_k != int(ev.get("leave_k", eval_leave_k))
        or eval_relevance_threshold != float(ev.get("relevance_threshold", eval_relevance_threshold))
        or eval_max_windows != int(ev.get("max_windows", eval_max_windows))
    )
    if params_changed:
        st.warning(
            f"Showing cached results (N={ev.get('top_k')}, K={ev.get('leave_k')}, "
            f"liked≥{ev.get('relevance_threshold', 3.5)}★). "
            f"Hit **Re-run Evaluation** to apply your new settings."
        )

    # ── How the evaluation works ───────────────────────────────────────────────
    stored_leave_k = ev.get("leave_k", eval_leave_k)
    stored_top_k   = ev.get("top_k",   eval_top_k)
    st.markdown(
        '<div style="background:#111;border:1px solid #1e1e1e;border-left:4px solid #e50914;'
        'border-radius:8px;padding:18px 20px;margin-bottom:28px">'
        '<div style="font-family:Cinzel,serif;font-size:0.75rem;color:#e50914;letter-spacing:2px;'
        'margin-bottom:8px">HOW WE EVALUATE</div>'
        '<p style="color:#ccc;font-size:0.88rem;line-height:1.7;margin:0">'
        f'<b>Sliding-window leave-k-out</b> — for each user we slide a window of '
        f'<b>K={stored_leave_k}</b> movies backwards through their rating history, '
        f'creating up to <b>{ev.get("max_windows", 3)}</b> non-overlapping train/test splits. '
        f'Each split uses everything before the window as training. '
        f'Metrics are averaged across windows per user, then across users. '
        f'A hidden movie is a <b>hit</b> only if the user liked it (rating ≥ threshold). '
        f'<b>Precision</b> = liked hits ÷ liked held-out (e.g. user liked 4 of {stored_leave_k} held-out, '
        f'we recommended 3 → 3/4). '
        '<b>Graded NDCG</b> uses actual ratings as gains, rewarding higher-ranked beloved films more.'
        '</p></div>',
        unsafe_allow_html=True,
    )

    if not ev:
        st.info(
            "No evaluation results yet. Click **Re-run Evaluation** in the sidebar "
            "(requires processed data and embeddings to exist)."
        )

    # ── Metric explanations ────────────────────────────────────────────────────
    k = ev.get("top_k", 10)
    _dk = ev.get("dislike_threshold", 2.5)
    _lk = ev.get("leave_k", 5)
    METRICS = [
        {
            "key":   "precision_at_k",
            "label": "Precision (Liked Caught / Liked Held-Out)",
            "icon":  "🎯",
            "color": "#e50914",
            "what":  f"K = {_lk} movies are held out per user. Of those, however many the user liked "
                     f"(rated ≥ threshold) form the denominator. Precision = liked hits ÷ liked held-out. "
                     f"Example: user has 4 liked movies among their {_lk} held-out; we recommended 3 of them → 3/4 = 0.75.",
            "good":  "Higher = we recommended a larger share of the movies the user actually liked in their held-out set.",
            "range": "0 → 1. Denominator varies per user (their liked held-out count, not K).",
            "fmt":   "0-1",
        },
        {
            "key":   "ndcg_at_k",
            "label": f"NDCG@{k} (Binary)",
            "icon":  "📈",
            "color": "#c9a227",
            "what":  f"Are liked movies near the top of the list? A liked hit at rank 1 scores more "
                     f"than the same hit at rank {k}. Only verifiable positions (test-set movies) are scored.",
            "good":  "Higher = liked movies ranked first, not buried at the bottom.",
            "range": "0 → 1. Normalised against the ideal ranking for that user.",
            "fmt":   "0-1",
        },
        {
            "key":   "graded_ndcg_at_k",
            "label": f"Graded NDCG@{k} (Rating-Weighted)",
            "icon":  "🌡️",
            "color": "#c9a227",
            "what":  f"Like NDCG, but gain is the actual rating (0–5) ÷ 5 instead of binary 0/1. "
                     f"A rec of a 5★ film scores twice as much as a 2.5★ one at the same rank. "
                     f"Gives partial credit and is more lenient toward near-liked recommendations.",
            "good":  "Higher = highly-rated held-out movies appear early in the list.",
            "range": "0 → 1. Will generally be higher than binary NDCG for the same model.",
            "fmt":   "0-1",
        },
        {
            "key":   "hit_rate_at_k",
            "label": f"Hit Rate@{k}",
            "icon":  "✅",
            "color": "#1a7a40",
            "what":  f"What fraction of users received at least one liked movie in their top {k}?",
            "good":  "Higher = more users get at least one good recommendation.",
            "range": "0 → 1. Lenient metric — even one hit per user counts as success.",
            "fmt":   "0-1",
        },
        {
            "key":   "pairwise_rank_acc_at_k",
            "label": "Pairwise Ranking Accuracy",
            "icon":  "⚖️",
            "color": "#7b2fbe",
            "what":  "For every pair of liked held-out movies (A rated higher than B), "
                     "what fraction does the model rank A above B? "
                     "Movies not in the top-N are treated as ranked last. "
                     "Computed only over users with ≥ 2 liked held-out movies at different ratings.",
            "good":  "Higher = the model's ordering within liked movies matches the user's preference strength.",
            "range": "0 → 1. 0.5 = random ordering; 1.0 = perfect preference-consistent ranking.",
            "fmt":   "0-1",
        },
        {
            "key":   "dislike_rate_at_k",
            "label": f"Dislike Rate@{k}",
            "icon":  "👎",
            "color": "#c45e0a",
            "what":  f"Of the recommended movies the user actually watched (verified in test set), "
                     f"what fraction did they actively dislike (rated ≤ {_dk}★)?",
            "good":  "Lower = fewer bad recommendations that the user watched and disliked.",
            "range": "0 → 1. 0 means none of the verified recommendations were disliked.",
            "fmt":   "0-1",
        },
        {
            "key":   "coverage_pct_at_k",
            "label": "Catalog Coverage",
            "icon":  "🗺️",
            "color": "#0a7a70",
            "what":  "Percentage of the full movie catalog that gets recommended to at least one user.",
            "good":  "Higher = more diverse recommendations; low coverage suggests popularity bias.",
            "range": "0 → 100%. A high score means the model explores beyond the same blockbusters.",
            "fmt":   "pct",
        },
    ]

    # ── Metric cards ──────────────────────────────────────────────────────────
    for m in METRICS:
        val = ev.get(m["key"], None)
        if val is None:
            continue
        if m["fmt"] == "pct":
            bar_w = min(100, max(0, int(val)))
            val_s = f"{val:.2f}%"
        else:
            # For dislike_rate, lower is better — invert the bar fill direction
            raw_w = min(100, max(0, int(val * 100)))
            bar_w = (100 - raw_w) if m["key"] == "dislike_rate_at_k" else raw_w
            val_s = f"{val:.4f}"

        st.markdown(
            f'<div style="background:#111;border:1px solid #1e1e1e;border-radius:10px;'
            f'padding:18px 22px;margin-bottom:14px">'
            # Header row
            f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">'
            f'<div style="display:flex;align-items:center;gap:10px">'
            f'<span style="font-size:1.4rem">{m["icon"]}</span>'
            f'<span style="font-family:Cinzel,serif;font-size:0.95rem;color:#f0f0f0;font-weight:700">'
            f'{m["label"]}</span>'
            f'</div>'
            f'<span style="font-size:1.6rem;font-weight:800;color:{m["color"]}">{val_s}</span>'
            f'</div>'
            # Progress bar
            f'<div style="background:#1a1a1a;border-radius:4px;height:6px;margin-bottom:14px">'
            f'<div style="background:{m["color"]};border-radius:4px;height:6px;width:{bar_w}%">'
            f'</div></div>'
            # Explanations
            f'<p style="color:#ccc;font-size:0.83rem;margin:0 0 6px"><b>What it measures:</b> {m["what"]}</p>'
            f'<p style="color:#888;font-size:0.8rem;margin:0 0 4px">{m["good"]}</p>'
            f'<p style="color:#555;font-size:0.75rem;margin:0"><i>Range: {m["range"]}</i></p>'
            + (
                f'<p style="color:#333;font-size:0.72rem;margin-top:6px">'
                f'Averaged over {ev.get("pairwise_rank_acc_users", "?")} users '
                f'(those with ≥ 2 liked held-out movies at different ratings).</p>'
                if m["key"] == "pairwise_rank_acc_at_k" else ""
            ) +
            f'</div>',
            unsafe_allow_html=True,
        )

    if ev and "coverage_pct_at_k" not in ev:
        st.info("Catalog Coverage requires a fresh evaluation run — click **Re-run Evaluation**.")

    # ── Summary stats ─────────────────────────────────────────────────────────
    if ev:
        st.markdown("---")
        st.markdown(
            '<p style="color:#444;font-size:0.82rem">Evaluation dataset:</p>',
            unsafe_allow_html=True,
        )
        sc1, sc2, sc3, sc4, sc5 = st.columns(5)
        sc1.metric("Users evaluated",    ev.get("users_evaluated", "—"))
        sc2.metric("Total users",        ev.get("users_total", "—"))
        sc3.metric("K (held out)",       ev.get("leave_k", "—"))
        sc4.metric("Max windows",        ev.get("max_windows", "—"))
        sc5.metric("Avg windows / user", f'{ev.get("avg_windows_per_user", 0):.1f}')

    # ── Live user count preview ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div style="font-family:Cinzel,serif;font-size:0.75rem;color:#e50914;'
        'letter-spacing:2px;margin-bottom:12px">CURRENT PARAMETER PREVIEW</div>',
        unsafe_allow_html=True,
    )
    ratings_csv = str(PROJECT_ROOT / "data" / "processed" / "ratings_clean.csv")
    if Path(ratings_csv).exists():
        with st.spinner("Counting eligible users…"):
            eligible, enough_hist, total_u = count_eligible_users(
                ratings_csv,
                leave_k=eval_leave_k,
                min_train_ratings=eval_min_train,
                min_relevant_test=eval_min_rel,
                relevance_threshold=eval_relevance_threshold,
            )
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("Total users", f"{total_u:,}")
        pc2.metric(f"Have ≥{eval_leave_k + 5} ratings", f"{enough_hist:,}",
                   help=f"Users with enough history to leave {eval_leave_k} out and still have 5 training ratings.")
        pc3.metric(
            f"Eligible (≥{eval_min_rel} liked in test)",
            f"{eligible:,}",
            delta=f"{eligible/total_u*100:.1f}% of all" if total_u else None,
            help=f"Users whose held-out set contains ≥{eval_min_rel} movie(s) rated ≥{eval_relevance_threshold}★.",
        )
        st.markdown(
            f'<p style="color:#333;font-size:0.75rem;margin-top:4px">'
            f'With these settings: N={eval_top_k}, K={eval_leave_k}, '
            f'min_liked_in_test={eval_min_rel}, liked≥{eval_relevance_threshold}★ — '
            f'<b style="color:#888">{eligible:,}</b> users would be included in the evaluation.'
            f'</p>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Ratings file not found — run the data pipeline first to see user counts.")

    # ── Sensitivity Analysis ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div style="font-family:Cinzel,serif;font-size:0.75rem;color:#e50914;'
        'letter-spacing:2px;margin-bottom:12px">SENSITIVITY ANALYSIS</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#555;font-size:0.82rem;margin-bottom:16px">'
        'Vary one parameter across a range while holding the others fixed at your '
        'current sidebar values. Results are cached — re-running the same config is instant.</p>',
        unsafe_allow_html=True,
    )

    _param_labels = {
        "top_k":               "N — recommendations shown",
        "leave_k":             "K — movies held out",
        "min_relevant_test":   "Min liked movies in test set",
        "relevance_threshold": "Liked threshold (rating ≥)",
    }
    sa_col1, sa_col2 = st.columns([3, 1])
    with sa_col1:
        vary_param = st.selectbox(
            "Parameter to vary",
            options=list(_param_labels.keys()),
            format_func=lambda x: _param_labels[x],
            label_visibility="collapsed",
        )
    with sa_col2:
        run_sa = st.button("Run Analysis", use_container_width=True, key="run_sensitivity")

    if run_sa:
        _rc = str(PROJECT_ROOT / "data" / "processed" / "ratings_clean.csv")
        if not Path(_rc).exists():
            st.error("Ratings file not found — run the data pipeline first.")
        elif not EMBEDDINGS_PATH.exists():
            st.error("Embeddings not found — build them first.")
        else:
            with st.spinner(f"Running sensitivity sweep on '{_param_labels[vary_param]}'…"):
                _sens_df = run_sensitivity_analysis(
                    ratings_path_str=_rc,
                    embeddings_path_str=str(EMBEDDINGS_PATH),
                    vary_param=vary_param,
                    base_top_k=eval_top_k,
                    base_leave_k=eval_leave_k,
                    base_min_train=eval_min_train,
                    base_min_rel=eval_min_rel,
                    base_threshold=eval_relevance_threshold,
                    base_max_windows=eval_max_windows,
                )
            if "sensitivity_results" not in st.session_state:
                st.session_state.sensitivity_results = {}
            st.session_state.sensitivity_results[vary_param] = {
                "df": _sens_df,
                "params": dict(top_k=eval_top_k, leave_k=eval_leave_k,
                               min_rel=eval_min_rel, threshold=eval_relevance_threshold),
            }

    _sr = st.session_state.get("sensitivity_results", {})
    if vary_param in _sr and not _sr[vary_param]["df"].empty:
        _entry = _sr[vary_param]
        _df    = _entry["df"]
        _p     = _entry["params"]
        st.caption(
            f"Baseline used — N={_p['top_k']}, K={_p['leave_k']}, "
            f"min_liked={_p['min_rel']}, liked≥{_p['threshold']}★"
        )
        _metric_cols = [c for c in _df.columns if c != "Users Evaluated"]
        st.markdown("**Metric scores vs parameter value:**")
        st.line_chart(_df[_metric_cols], height=320)
        st.markdown("**Users included in each run:**")
        st.line_chart(_df[["Users Evaluated"]], height=180)
        with st.expander("Raw numbers"):
            st.dataframe(_df.reset_index().style.format(
                {c: "{:.4f}" for c in _metric_cols} | {"Users Evaluated": "{:.0f}"}
            ))


# ── Module-level page objects (url_path must match <a href> links in page_home) ─
_p_home       = st.Page(page_home,       title="Home",                    icon="🏠")
_p_browse     = st.Page(page_browse,     title="Browse",                  icon="🎬", url_path="browse")
_p_my_ratings = st.Page(page_my_ratings, title=f"My Ratings ({n_rated})", icon="⭐", url_path="my-ratings")
_p_for_you    = st.Page(page_for_you,    title="For You",                 icon="🎯", url_path="for-you")
_p_evaluation = st.Page(page_evaluation, title="Evaluation",              icon="📊", url_path="evaluation")


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED SIDEBAR (logo + stats + OMDB key — appears below the nav links)
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;padding:12px 0 8px">'
        '<div style="font-size:2.4rem">🎬</div>'
        '<div style="font-family:Cinzel,serif;font-size:1.3rem;font-weight:900;'
        'color:#e50914;letter-spacing:3px;line-height:1.1">CINE</div>'
        '<div style="font-family:Cinzel,serif;font-size:1.3rem;font-weight:400;'
        'color:#f5c518;letter-spacing:5px;line-height:1.1">MATCH</div>'
        '<div style="color:#333;font-size:0.55rem;letter-spacing:4px;margin-top:4px">EST. 2026</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    col_a, col_b = st.columns(2)
    col_a.metric("Rated", n_rated)
    col_b.metric("Liked ★★★★+", n_liked)

    if n_rated > 0:
        avg_u = sum(v["rating"] for v in st.session_state.ratings.values()) / n_rated
        st.markdown(
            f'<div style="text-align:center;color:#f5c518;font-size:0.82rem;padding:4px 0">'
            f'Your avg: {avg_u:.1f} ★</div>',
            unsafe_allow_html=True,
        )
        if st.button("Clear All Ratings", use_container_width=True):
            st.session_state.ratings = {}
            st.session_state.recs = None
            st.rerun()

    st.markdown("---")



# ═══════════════════════════════════════════════════════════════════════════════
# NAVIGATION — runs the current page
# ═══════════════════════════════════════════════════════════════════════════════

pg = st.navigation([_p_home, _p_browse, _p_my_ratings, _p_for_you, _p_evaluation])
pg.run()
