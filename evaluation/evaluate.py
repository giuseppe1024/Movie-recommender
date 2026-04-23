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


def _ndcg_at_k(recommended_ids: list[int], relevant_ids: set[int], k: int) -> float:
	gains = np.array([1.0 if mid in relevant_ids else 0.0 for mid in recommended_ids[:k]])
	if gains.size == 0:
		return 0.0
	discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
	dcg = float(np.sum(gains * discounts))

	ideal_len = min(k, len(relevant_ids))
	if ideal_len == 0:
		return 0.0
	ideal_discounts = 1.0 / np.log2(np.arange(2, ideal_len + 2))
	idcg = float(np.sum(ideal_discounts))
	return dcg / idcg if idcg > 0 else 0.0


def _mrr_at_k(recommended_ids: list[int], relevant_ids: set[int], k: int) -> float:
	for rank, movie_id in enumerate(recommended_ids[:k], start=1):
		if movie_id in relevant_ids:
			return 1.0 / rank
	return 0.0


def evaluate_temporal_leave_k_out(
	ratings_path: str | Path | None = None,
	embeddings_path: str | Path | None = None,
	leave_k: int = 5,
	top_k: int = 50,
	relevance_threshold: float = 4.0,
	min_train_ratings: int = 5,
	weighted: bool = True,
) -> tuple[dict, pd.DataFrame]:
	"""Run temporal leave-k-out evaluation over users.

	For each user, the most recent ``leave_k`` interactions are held out as test.
	All earlier interactions are used as training history.
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

	for user_id, user_hist in ratings.groupby("userId", sort=False):
		num_users_total += 1
		if len(user_hist) < (leave_k + min_train_ratings):
			continue

		user_hist = user_hist.sort_values("timestamp")
		train = user_hist.iloc[:-leave_k].copy()
		test = user_hist.iloc[-leave_k:].copy()

		if len(train) < min_train_ratings:
			continue

		relevant_test = set(test.loc[test["rating"] >= relevance_threshold, "movieId"].tolist())
		if not relevant_test:
			continue

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
			# No high-rated training items for this user.
			continue

		rec_ids = recs["movieId"].tolist()
		hits = len(set(rec_ids).intersection(relevant_test))
		recommended_catalog_ids.update(rec_ids)

		precision = hits / top_k if top_k > 0 else 0.0
		recall = hits / len(relevant_test) if relevant_test else 0.0
		hit_rate = 1.0 if hits > 0 else 0.0
		ndcg = _ndcg_at_k(rec_ids, relevant_test, top_k)
		mrr = _mrr_at_k(rec_ids, relevant_test, top_k)

		per_user_rows.append(
			{
				"userId": int(user_id),
				"train_count": int(len(train)),
				"test_count": int(len(test)),
				"relevant_test_count": int(len(relevant_test)),
				"hits_at_k": int(hits),
				"precision_at_k": float(precision),
				"recall_at_k": float(recall),
				"hit_rate_at_k": float(hit_rate),
				"ndcg_at_k": float(ndcg),
				"mrr_at_k": float(mrr),
			}
		)

	per_user = pd.DataFrame(per_user_rows)
	if per_user.empty:
		summary = {
			"users_total": int(num_users_total),
			"users_evaluated": 0,
			"leave_k": int(leave_k),
			"top_k": int(top_k),
			"precision_at_k": 0.0,
			"recall_at_k": 0.0,
			"hit_rate_at_k": 0.0,
			"ndcg_at_k": 0.0,
			"mrr_at_k": 0.0,
			"coverage_at_k": 0.0,
			"coverage_pct_at_k": 0.0,
		}
		return summary, per_user

	coverage = len(recommended_catalog_ids) / len(movie_ids) if len(movie_ids) > 0 else 0.0

	summary = {
		"users_total": int(num_users_total),
		"users_evaluated": int(len(per_user)),
		"leave_k": int(leave_k),
		"top_k": int(top_k),
		"precision_at_k": float(per_user["precision_at_k"].mean()),
		"recall_at_k": float(per_user["recall_at_k"].mean()),
		"hit_rate_at_k": float(per_user["hit_rate_at_k"].mean()),
		"ndcg_at_k": float(per_user["ndcg_at_k"].mean()),
		"mrr_at_k": float(per_user["mrr_at_k"].mean()),
		"coverage_at_k": float(coverage),
		"coverage_pct_at_k": float(coverage * 100.0),
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
	parser.add_argument("--relevance-threshold", type=float, default=4.0)
	parser.add_argument("--min-train-ratings", type=int, default=5)
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
