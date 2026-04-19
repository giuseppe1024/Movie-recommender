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

[data-testid="stSidebar"] { background: #0d0d0d !important; border-right: 1px solid #1c1c1c; }
[data-testid="stSidebar"] * { color: #d8d8d8 !important; }

/* Nav links styling */
[data-testid="stSidebarNavLink"] {
    border-radius: 8px !important;
    margin: 2px 0 !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}
[data-testid="stSidebarNavLink"][aria-selected="true"] {
    background: rgba(229,9,20,0.15) !important;
    border-left: 3px solid #e50914 !important;
    color: #fff !important;
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
                poster_url = (omdb or {}).get("poster", "")
                if poster_url and poster_url not in ("", "N/A"):
                    st.image(poster_url, use_container_width=True)
                else:
                    st.markdown(poster_placeholder(mid, dtitle, year),
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
        poster_url = (omdb or {}).get("poster","")

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
                mc1.metric("Precision@10",  f'{ev.get("precision_at_k", 0):.3f}')
                mc2.metric("Recall@10",     f'{ev.get("recall_at_k", 0):.3f}')
                mc3.metric("NDCG@10",       f'{ev.get("ndcg_at_k", 0):.3f}')
                mc4.metric("Hit Rate@10",   f'{ev.get("hit_rate_at_k", 0):.3f}')
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
        omdb    = fetch_movie_meta(dtitle, None, tmdb_id)
        poster_url = (omdb or {}).get("poster", "")

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
    with st.sidebar:
        st.markdown("**Options**")
        if st.button("Re-run Evaluation", use_container_width=True,
                     help="Re-runs the full temporal leave-k-out evaluation. May take a minute."):
            with st.spinner("Running evaluation…"):
                try:
                    from evaluation.evaluate import evaluate_temporal_leave_k_out
                    summary, _ = evaluate_temporal_leave_k_out(
                        ratings_path=str(PROJECT_ROOT / "data" / "processed" / "ratings_clean.csv"),
                        embeddings_path=str(EMBEDDINGS_PATH),
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

    # ── How the evaluation works ───────────────────────────────────────────────
    st.markdown(
        '<div style="background:#111;border:1px solid #1e1e1e;border-left:4px solid #e50914;'
        'border-radius:8px;padding:18px 20px;margin-bottom:28px">'
        '<div style="font-family:Cinzel,serif;font-size:0.75rem;color:#e50914;letter-spacing:2px;'
        'margin-bottom:8px">HOW WE EVALUATE</div>'
        '<p style="color:#ccc;font-size:0.88rem;line-height:1.7;margin:0">'
        '<b>Temporal leave-k-out</b> — for each user in the dataset, we hide their '
        '<b>5 most recent</b> movie interactions and ask the model: given this user\'s '
        'earlier watch history, what would you recommend? We then check how many of the '
        'hidden movies appear in the top-10 recommendations. This mimics real-world usage — '
        'the model only sees past behaviour and must predict future taste.'
        '</p></div>',
        unsafe_allow_html=True,
    )

    # ── Metric explanations ────────────────────────────────────────────────────
    METRICS = [
        {
            "key":   "precision_at_k",
            "label": "Precision@10",
            "icon":  "🎯",
            "color": "#e50914",
            "what":  "Of the 10 movies recommended, how many did the user actually like?",
            "good":  "Higher = fewer irrelevant recommendations in the top 10.",
            "range": "0 → 1. A score of 0.10 means 1 in 10 recommended movies is relevant.",
        },
        {
            "key":   "recall_at_k",
            "label": "Recall@10",
            "icon":  "📡",
            "color": "#1a6eb5",
            "what":  "Of all the movies the user would enjoy, how many made it into the top 10?",
            "good":  "Higher = the model misses fewer movies the user would have loved.",
            "range": "0 → 1. Hard to max out — a user may love 50 films but only 10 are shown.",
        },
        {
            "key":   "ndcg_at_k",
            "label": "NDCG@10",
            "icon":  "📈",
            "color": "#c9a227",
            "what":  "Are relevant movies near the top of the list? NDCG rewards ranking quality — "
                     "a liked movie at position 1 scores more than the same movie at position 9.",
            "good":  "Higher = relevant movies ranked first, not buried at the bottom.",
            "range": "0 → 1. Normalised against the ideal ranking for that user.",
        },
        {
            "key":   "hit_rate_at_k",
            "label": "Hit Rate@10",
            "icon":  "✅",
            "color": "#1a7a40",
            "what":  "What fraction of users received at least one relevant movie in their top 10?",
            "good":  "Higher = more users get at least one good recommendation (nobody is left out).",
            "range": "0 → 1. The most lenient metric — even one hit counts as a success.",
        },
    ]

    if EVAL_RESULTS_PATH.exists():
        try:
            ev = json.loads(EVAL_RESULTS_PATH.read_text())
        except Exception:
            ev = {}
    else:
        ev = {}

    if not ev:
        st.info(
            "No evaluation results yet. Click **Re-run Evaluation** in the sidebar "
            "(requires processed data and embeddings to exist)."
        )

    # ── Metric cards ──────────────────────────────────────────────────────────
    for m in METRICS:
        val   = ev.get(m["key"], None)
        bar_w = int(val * 100) if val is not None else 0
        val_s = f"{val:.4f}" if val is not None else "—"

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
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Summary stats ─────────────────────────────────────────────────────────
    if ev:
        st.markdown("---")
        st.markdown(
            '<p style="color:#444;font-size:0.82rem">Evaluation dataset:</p>',
            unsafe_allow_html=True,
        )
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Users evaluated", ev.get("users_evaluated", "—"))
        sc2.metric("Total users",     ev.get("users_total", "—"))
        sc3.metric("Leave-k",         ev.get("leave_k", "—"))


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

pg = st.navigation([
    st.Page(page_browse,     title="Browse",                   icon="🎬"),
    st.Page(page_my_ratings, title=f"My Ratings ({n_rated})",  icon="⭐"),
    st.Page(page_for_you,    title="For You",                   icon="🎯"),
    st.Page(page_evaluation, title="Evaluation",                icon="📊"),
])
pg.run()
