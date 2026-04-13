# Build movie embedding vectors from metadata using sentence transformers, contrastive learning, and cosine similarity

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sentence_transformers import SentenceTransformer

from sklearn.preprocessing import normalize


# ── Contrastive fine-tuning ───────────────────────────────────────────────────

#Determines which movies are rated highly by the same users
class ContrastivePairDataset:
    def __init__(self, ratings_df, movie_texts, min_corated=10):
        high = ratings_df[ratings_df["rating"] >= 4.0]
        # group highly-rated movieIds per user
        user_movies = high.groupby("userId")["movieId"].apply(list)

        pairs = set()
        for movies in user_movies:
            movies = [m for m in movies if m in movie_texts]
            for i in range(len(movies)):
                for j in range(i + 1, len(movies)):
                    a, b = sorted([movies[i], movies[j]])
                    pairs.add((a, b))

        # Keep only pairs that appear at least min_corated times across users
        from collections import Counter
        pair_counts = Counter()
        for movies in user_movies:
            movies = [m for m in movies if m in movie_texts]
            for i in range(len(movies)):
                for j in range(i + 1, len(movies)):
                    pair_counts[tuple(sorted([movies[i], movies[j]]))] += 1

        # Store pairs as (idx_a, idx_b) into the movie_ids list
        mid2idx = {mid: i for i, mid in enumerate(movie_texts.keys())}
        self.pairs = [
            (mid2idx[a], mid2idx[b])
            for (a, b), cnt in pair_counts.items()
            if cnt >= min_corated and a in mid2idx and b in mid2idx
        ]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


class ProjectionHead(nn.Module):
    #MLP Head on top of BERT
    def __init__(self, in_dim, hidden_dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def nt_xent_loss(z1, z2, temperature=0.07):
    #Loss function
    batch = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)                    # (2B, D)
    sim = torch.mm(z, z.T) / temperature               # (2B, 2B)

    # Mask out self-similarity on the diagonal
    mask = torch.eye(2 * batch, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float("-inf"))

    # Positive indices: for row i the positive is i+B, for row i+B it is i
    labels = torch.cat([
        torch.arange(batch, 2 * batch, device=z.device),
        torch.arange(0, batch, device=z.device),
    ])
    return F.cross_entropy(sim, labels)


def contrastive_finetune(
    pairs,
    all_base_embeddings,
    device,
    epochs=3,
    batch_size=64,
    neg_per_pair=32,
    lr=1e-3,
    temperature=0.07,
):
   #Fine tunes on top of pre-trained BERT model
    all_emb_tensor = torch.tensor(all_base_embeddings, dtype=torch.float32)
    n_all = all_emb_tensor.shape[0]
    emb_dim = all_emb_tensor.shape[1]

    # Map pair texts → indices into all_base_embeddings via text2emb built outside
    # (pairs already carry pre-encoded indices passed in as index tuples)
    pair_a_idx, pair_b_idx = zip(*pairs)
    pair_a_idx = torch.tensor(pair_a_idx, dtype=torch.long)
    pair_b_idx = torch.tensor(pair_b_idx, dtype=torch.long)

    head = ProjectionHead(emb_dim).to(device)
    optimiser = torch.optim.Adam(head.parameters(), lr=lr)

    n = len(pairs)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        pair_a_idx = pair_a_idx[perm]
        pair_b_idx = pair_b_idx[perm]
        total_loss = 0.0
        steps = 0
        for start in range(0, n, batch_size):
            a_idx = pair_a_idx[start:start + batch_size]
            b_idx = pair_b_idx[start:start + batch_size]
            if a_idx.shape[0] < 2:
                continue

            # Sample random negatives from the full catalogue
            neg_idx = torch.randint(0, n_all, (neg_per_pair * a_idx.shape[0],))
            # Combine: positives first, then negatives
            all_idx = torch.cat([a_idx, b_idx, neg_idx])
            vecs = all_emb_tensor[all_idx].to(device)

            batch_sz = a_idx.shape[0]
            a_proj = head(vecs[:batch_sz])                     # (B, D)
            b_proj = head(vecs[batch_sz:2 * batch_sz])         # (B, D)
            neg_proj = head(vecs[2 * batch_sz:])               # (B*neg, D)

            # Build augmented z1/z2: each row's positive is its pair partner;
            # all other rows (including negatives) act as negatives in NT-Xent
            z = torch.cat([a_proj, b_proj, neg_proj], dim=0)  # (2B + B*neg, D)
            z = F.normalize(z, dim=-1)
            total = z.shape[0]
            sim = torch.mm(z, z.T) / temperature
            mask = torch.eye(total, dtype=torch.bool, device=device)
            sim.masked_fill_(mask, float("-inf"))

            # Positives: row i's positive is row i+B; row i+B's positive is row i
            labels = torch.cat([
                torch.arange(batch_sz, 2 * batch_sz, device=device),
                torch.arange(0, batch_sz, device=device),
            ])
            # Only compute loss over the 2B pair rows (not the pure negatives)
            loss = F.cross_entropy(sim[:2 * batch_sz], labels)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
            steps += 1
        print(f"  Epoch {epoch + 1}/{epochs}  loss={total_loss / max(steps, 1):.4f}")

    return head

def build_embeddings(
    processed_dir="data/processed",
    output_dir="embeddings",
    model_name="all-MiniLM-L6-v2",
    finetune=True,
    finetune_epochs=3,
    min_corated=10,
    batch_size=64,
):

    # Loads data, encodes with pretrained, fine tunes w/ contrastive learning, and makes a cav
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    movies = pd.read_csv(processed_dir / "embedding_movies.csv")
    ratings = pd.read_csv(processed_dir / "ratings_clean.csv")

    # Build {movieId: text} lookup
    movie_texts = dict(zip(movies["movieId"], movies["movies_text"].fillna("")))
    movie_ids = movies["movieId"].tolist()
    texts = [movie_texts[mid] for mid in movie_ids]

    # ── Base encoding ─────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading sentence-transformer '{model_name}' on {device} …")
    base_model = SentenceTransformer(model_name, device=device)

    print(f"Encoding {len(texts)} movies …")
    base_embeddings = base_model.encode(
        texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True
    )  # shape: (N, base_dim)

    # ── Contrastive fine-tuning ───────────────────────────────────────────────
    if finetune:
        print("Building contrastive pairs from co-rated movies …")
        dataset = ContrastivePairDataset(ratings, movie_texts, min_corated=min_corated)
        print(f"  Found {len(dataset)} positive pairs (min_corated={min_corated})")

        if len(dataset) < 2:
            print("  Too few pairs for fine-tuning – skipping.")
            final_embeddings = base_embeddings
        else:
            head = contrastive_finetune(
                pairs=dataset.pairs,
                all_base_embeddings=base_embeddings,
                device=device,
                epochs=finetune_epochs,
                batch_size=batch_size,
            )
            # Project all movie embeddings through the trained head
            print("Projecting all movie embeddings through fine-tuned head …")
            head.eval()
            all_base = torch.tensor(base_embeddings, dtype=torch.float32).to(device)
            with torch.no_grad():
                chunks = torch.split(all_base, 512)
                projected = torch.cat([head(c) for c in chunks], dim=0)
            final_embeddings = projected.cpu().numpy()
    else:
        final_embeddings = base_embeddings

    # ── L2-normalise for cosine similarity via dot product ────────────────────
    final_embeddings = normalize(final_embeddings, norm="l2").astype(np.float32)

    # ── Save ──────────────────────────────────────────────────────────────────
    emb_path = output_dir / "movie_embeddings.csv"

    dim = final_embeddings.shape[1]
    emb_df = pd.DataFrame(final_embeddings, columns=[f"emb_{i}" for i in range(dim)])
    emb_df.insert(0, "movieId", movie_ids)
    emb_df.insert(1, "title", movies["title"].tolist())
    emb_df.to_csv(emb_path, index=False)

    print(f"Saved embeddings → {emb_path}  shape={final_embeddings.shape}")
    return final_embeddings, emb_df


if __name__ == "__main__":
    build_embeddings()
