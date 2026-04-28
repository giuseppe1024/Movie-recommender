"""Evaluate recommendation quality using temporal leave-k-out split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.recommend import build_user_query_vector, load_embedding_table


def _resolve_default_paths(base_dir: Path) -> tuple[Path, Path]:
	ratings_candidates = [
		base_dir / "data" / "processed" / "ratings_clean.csv",
		base_dir / "data" / "ratings.csv",
	]
	embeddings_candidates = [
		base_dir / "embeddings" / "movie_embeddings.csv",
	]

	ratings_path = next((p for p in ratings_candidates if p.exists()), ratings_candidates[0])
	embeddings_path = next(
		(p for p in embeddings_candidates if p.exists()),
		embeddings_candidates[0],
	)
	return ratings_path, embeddings_path


def _recommend_with_preloaded_embeddings(
	user_train: pd.DataFrame,
	emb_matrix: np.ndarray,
	movieid_to_idx: dict,
	movie_ids: np.ndarray,
	titles: np.ndarray,
	min_rating: float,
	top_k: int,
	weighted: bool,
) -> pd.DataFrame:
	query_vec, _ = build_user_query_vector(
		user_ratings=user_train,
		emb_matrix=emb_matrix,
		movieid_to_idx=movieid_to_idx,
		min_rating=min_rating,
		weighted=weighted,
	)

	sims = emb_matrix @ query_vec
	already_rated = set(user_train["movieId"].tolist())
	rated_mask = np.array([mid in already_rated for mid in movie_ids])
	sims[rated_mask] = -np.inf

	top_idx = np.argsort(-sims)[:top_k]
	return pd.DataFrame(
		{
			"movieId": movie_ids[top_idx],
			"title": titles[top_idx],
			"cosine_similarity": sims[top_idx],
		}
	).reset_index(drop=True)


def _ndcg_at_k(
	recommended_ids: list[int],
	relevant_ids: set[int],
	k: int,
	watched_ids: set[int] | None = None,
) -> float:
	# If watched_ids supplied, only score positions where user actually watched the movie.
	# This avoids penalising recommendations we can't verify (user never saw them).
	if watched_ids is not None:
		candidate_ids = [mid for mid in recommended_ids[:k] if mid in watched_ids]
		ideal_n = len(relevant_ids & watched_ids)
	else:
		candidate_ids = list(recommended_ids[:k])
		ideal_n = len(relevant_ids)

	gains = np.array([1.0 if mid in relevant_ids else 0.0 for mid in candidate_ids])
	if gains.size == 0:
		return 0.0
	discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
	dcg = float(np.sum(gains * discounts))

	ideal_len = min(k, ideal_n)
	if ideal_len == 0:
		return 0.0
	ideal_discounts = 1.0 / np.log2(np.arange(2, ideal_len + 2))
	idcg = float(np.sum(ideal_discounts))
	return dcg / idcg if idcg > 0 else 0.0


def _pairwise_rank_accuracy(
	recommended_ids: list[int],
	test_ratings: dict[int, float],
	relevance_threshold: float,
	k: int,
) -> float:
	"""Fraction of liked-movie pairs (i, j) with rating(i) > rating(j) where the
	model ranks movie i above movie j.

	Movies outside the top-K are treated as ranked at position K+1 (penalised).
	Pairs where both movies are unranked (both get K+1) are skipped — no signal.
	Returns np.nan when fewer than 2 distinct-rating liked movies exist in the test set.
	"""
	liked = {mid: r for mid, r in test_ratings.items() if r >= relevance_threshold}
	if len(liked) < 2:
		return np.nan

	rec_rank = {mid: pos for pos, mid in enumerate(recommended_ids[:k], start=1)}
	correct = 0
	total = 0
	items = list(liked.items())
	for a in range(len(items)):
		for b in range(a + 1, len(items)):
			mid_hi, r_hi = items[a]
			mid_lo, r_lo = items[b]
			if r_hi < r_lo:
				mid_hi, mid_lo = mid_lo, mid_hi
				r_hi, r_lo = r_lo, r_hi
			if r_hi == r_lo:
				continue  # skip ties — no ground-truth ordering
			rank_hi = rec_rank.get(mid_hi, k + 1)
			rank_lo = rec_rank.get(mid_lo, k + 1)
			if rank_hi == rank_lo:
				continue  # both unranked — no signal
			total += 1
			if rank_hi < rank_lo:
				correct += 1
	return correct / total if total > 0 else np.nan


def _graded_ndcg_at_k(
	recommended_ids: list[int],
	test_ratings: dict[int, float],
	k: int,
	max_rating: float = 5.0,
) -> float:
	"""NDCG with graded relevance: gain = actual_rating / max_rating for test-set movies.

	Movies not in the test set get gain 0 (unverified).  The ideal ordering is the
	test-set movies sorted by rating descending.  This gives partial credit proportional
	to how much the user actually liked each recommended movie, rather than binary hit/miss.
	"""
	gains = np.array([
		test_ratings.get(mid, 0.0) / max_rating
		for mid in recommended_ids[:k]
	])
	discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
	dcg = float(np.sum(gains * discounts))

	ideal_gains = np.array(sorted(test_ratings.values(), reverse=True)[:k]) / max_rating
	if ideal_gains.size == 0 or ideal_gains.max() == 0:
		return 0.0
	ideal_discounts = 1.0 / np.log2(np.arange(2, ideal_gains.size + 2))
	idcg = float(np.sum(ideal_gains * ideal_discounts))
	return dcg / idcg if idcg > 0 else 0.0


def evaluate_temporal_leave_k_out(
	ratings_path: str | Path | None = None,
	embeddings_path: str | Path | None = None,
	leave_k: int = 5,
	top_k: int = 50,
	relevance_threshold: float = 3.5,
	dislike_threshold: float = 2.5,
	min_train_ratings: int = 5,
	min_relevant_test: int = 1,
	max_windows: int = 3,
	weighted: bool = True,
) -> tuple[dict, pd.DataFrame]:
	"""Sliding-window temporal leave-k-out evaluation.

	For each user we slide a window of size ``leave_k`` backwards through their
	rating history, creating multiple non-overlapping train/test splits.  Each
	split uses everything before the window as training; the window itself is the
	test set.  We keep sliding until the remaining training history would fall
	below ``min_train_ratings``.

	``max_windows`` caps how many windows we use per user (0 = no cap).
	Per-user metrics are averaged across all valid windows before aggregating
	across users, so every user contributes equally regardless of history length.
	"""

	base_dir = PROJECT_ROOT
	default_ratings, default_embeddings = _resolve_default_paths(base_dir)
	ratings_path = Path(ratings_path) if ratings_path else default_ratings
	embeddings_path = Path(embeddings_path) if embeddings_path else default_embeddings

	if not ratings_path.exists():
		raise FileNotFoundError(
			"Could not find ratings file. Provide --ratings-path or create one of: "
			f"{base_dir / 'data' / 'processed' / 'ratings_clean.csv'} or "
			f"{base_dir / 'data' / 'ratings.csv'}"
		)
	if not embeddings_path.exists():
		raise FileNotFoundError(
			"Could not find embeddings file. Provide --embeddings-path or create: "
			f"{base_dir / 'embeddings' / 'movie_embeddings.csv'}"
		)

	ratings = pd.read_csv(ratings_path)
	required_cols = {"userId", "movieId", "rating", "timestamp"}
	missing = required_cols.difference(ratings.columns)
	if missing:
		raise ValueError(
			f"ratings file must contain columns {sorted(required_cols)}; missing {sorted(missing)}"
		)

	ratings = ratings.sort_values(["userId", "timestamp"]).copy()

	_, emb_matrix, movieid_to_idx, movie_ids, titles = load_embedding_table(
		str(embeddings_path)
	)

	per_user_rows = []
	num_users_total = 0
	recommended_catalog_ids = set()
	window_limit = max_windows if max_windows > 0 else 10_000  # effectively unlimited

	for user_id, user_hist in ratings.groupby("userId", sort=False):
		num_users_total += 1
		if len(user_hist) < (leave_k + min_train_ratings):
			continue

		user_hist = user_hist.sort_values("timestamp").reset_index(drop=True)
		n = len(user_hist)

		# ── Slide the window backwards through the user's history ────────────
		window_rows: list[dict] = []

		for w in range(window_limit):
			test_end   = n - w * leave_k
			test_start = test_end - leave_k
			train_end  = test_start

			if test_start < 0 or train_end < min_train_ratings:
				break

			train = user_hist.iloc[:train_end]
			test  = user_hist.iloc[test_start:test_end]

			test_set_all      = set(test["movieId"].tolist())
			relevant_test     = set(test.loc[test["rating"] >= relevance_threshold, "movieId"].tolist())
			disliked_test     = set(test.loc[test["rating"] <= dislike_threshold,   "movieId"].tolist())
			test_ratings_dict = dict(zip(test["movieId"].tolist(), test["rating"].tolist()))

			if len(relevant_test) < min_relevant_test:
				continue  # window has too few liked movies — skip it, try next

			try:
				recs = _recommend_with_preloaded_embeddings(
					user_train=train[["movieId", "rating"]],
					emb_matrix=emb_matrix,
					movieid_to_idx=movieid_to_idx,
					movie_ids=movie_ids,
					titles=titles,
					min_rating=relevance_threshold,
					top_k=top_k,
					weighted=weighted,
				)
			except ValueError:
				continue  # no liked training movies for this window

			rec_ids = recs["movieId"].tolist()
			hits    = len(set(rec_ids).intersection(relevant_test))
			recommended_catalog_ids.update(rec_ids)

			precision    = hits / len(relevant_test) if relevant_test else 0.0
			hit_rate     = 1.0 if hits > 0 else 0.0
			ndcg         = _ndcg_at_k(rec_ids, relevant_test, top_k, watched_ids=test_set_all)
			graded_ndcg  = _graded_ndcg_at_k(rec_ids, test_ratings_dict, top_k)
			pra          = _pairwise_rank_accuracy(rec_ids, test_ratings_dict, relevance_threshold, top_k)
			watched_recs = {mid for mid in rec_ids if mid in test_set_all}
			dislike_rate = (
				len(watched_recs & disliked_test) / len(watched_recs)
				if watched_recs else 0.0
			)

			window_rows.append({
				"precision_at_k":         precision,
				"hit_rate_at_k":          hit_rate,
				"ndcg_at_k":              ndcg,
				"graded_ndcg_at_k":       graded_ndcg,
				"pairwise_rank_acc_at_k": pra,   # may be np.nan
				"dislike_rate_at_k":      dislike_rate,
				"relevant_test_count":    len(relevant_test),
				"hits_at_k":              hits,
			})

		if not window_rows:
			continue

		# ── Average across windows → one row per user ─────────────────────────
		def _wmean(key: str) -> float:
			return float(np.mean([r[key] for r in window_rows]))

		pra_vals = np.array([r["pairwise_rank_acc_at_k"] for r in window_rows], dtype=float)

		per_user_rows.append({
			"userId":                 int(user_id),
			"num_windows":            len(window_rows),
			"relevant_test_count":    _wmean("relevant_test_count"),
			"hits_at_k":              _wmean("hits_at_k"),
			"precision_at_k":         _wmean("precision_at_k"),
			"hit_rate_at_k":          _wmean("hit_rate_at_k"),
			"ndcg_at_k":              _wmean("ndcg_at_k"),
			"graded_ndcg_at_k":       _wmean("graded_ndcg_at_k"),
			"pairwise_rank_acc_at_k": float(np.nanmean(pra_vals)) if not np.all(np.isnan(pra_vals)) else np.nan,
			"dislike_rate_at_k":      _wmean("dislike_rate_at_k"),
		})

	per_user = pd.DataFrame(per_user_rows)
	if per_user.empty:
		summary = {
			"users_total": int(num_users_total),
			"users_evaluated": 0,
			"leave_k": int(leave_k),
			"top_k": int(top_k),
			"max_windows": int(max_windows),
			"avg_windows_per_user": 0.0,
			"min_relevant_test": int(min_relevant_test),
			"relevance_threshold": float(relevance_threshold),
			"dislike_threshold": float(dislike_threshold),
			"precision_at_k": 0.0,
			"hit_rate_at_k": 0.0,
			"ndcg_at_k": 0.0,
			"graded_ndcg_at_k": 0.0,
			"pairwise_rank_acc_at_k": 0.0,
			"pairwise_rank_acc_users": 0,
			"dislike_rate_at_k": 0.0,
			"coverage_at_k": 0.0,
			"coverage_pct_at_k": 0.0,
		}
		return summary, per_user

	coverage  = len(recommended_catalog_ids) / len(movie_ids) if len(movie_ids) > 0 else 0.0
	pra_values = per_user["pairwise_rank_acc_at_k"].to_numpy(dtype=float)

	summary = {
		"users_total":            int(num_users_total),
		"users_evaluated":        int(len(per_user)),
		"leave_k":                int(leave_k),
		"top_k":                  int(top_k),
		"max_windows":            int(max_windows),
		"avg_windows_per_user":   float(per_user["num_windows"].mean()),
		"min_relevant_test":      int(min_relevant_test),
		"relevance_threshold":    float(relevance_threshold),
		"dislike_threshold":      float(dislike_threshold),
		"precision_at_k":         float(per_user["precision_at_k"].mean()),
		"hit_rate_at_k":          float(per_user["hit_rate_at_k"].mean()),
		"ndcg_at_k":              float(per_user["ndcg_at_k"].mean()),
		"graded_ndcg_at_k":       float(per_user["graded_ndcg_at_k"].mean()),
		"pairwise_rank_acc_at_k": float(np.nanmean(pra_values)) if not np.all(np.isnan(pra_values)) else 0.0,
		"pairwise_rank_acc_users": int(np.sum(~np.isnan(pra_values))),
		"dislike_rate_at_k":      float(per_user["dislike_rate_at_k"].mean()),
		"coverage_at_k":          float(coverage),
		"coverage_pct_at_k":      float(coverage * 100.0),
	}
	return summary, per_user


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Evaluate recommender quality with temporal leave-k-out split"
	)
	parser.add_argument("--ratings-path", type=str, default=None)
	parser.add_argument("--embeddings-path", type=str, default=None)
	parser.add_argument("--leave-k", type=int, default=5)
	parser.add_argument("--top-k", type=int, default=50)
	parser.add_argument("--relevance-threshold", type=float, default=3.5)
	parser.add_argument("--min-train-ratings", type=int, default=5)
	parser.add_argument("--min-relevant-test", type=int, default=1)
	parser.add_argument("--max-windows", type=int, default=3)
	parser.add_argument("--unweighted", action="store_true")
	parser.add_argument("--save-per-user", type=str, default=None)
	args = parser.parse_args()

	summary, per_user = evaluate_temporal_leave_k_out(
		ratings_path=args.ratings_path,
		embeddings_path=args.embeddings_path,
		leave_k=args.leave_k,
		top_k=args.top_k,
		relevance_threshold=args.relevance_threshold,
		min_train_ratings=args.min_train_ratings,
		min_relevant_test=args.min_relevant_test,
		max_windows=args.max_windows,
		weighted=not args.unweighted,
	)

	print("Temporal leave-k-out evaluation summary")
	for key, val in summary.items():
		if isinstance(val, float):
			print(f"- {key}: {val:.4f}")
		else:
			print(f"- {key}: {val}")

	if args.save_per_user:
		out_path = Path(args.save_per_user)
		out_path.parent.mkdir(parents=True, exist_ok=True)
		per_user.to_csv(out_path, index=False)
		print(f"Saved per-user metrics to: {out_path}")


if __name__ == "__main__":
	main()
