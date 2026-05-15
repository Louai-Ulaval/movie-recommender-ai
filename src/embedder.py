"""
Computes sentence-transformer embeddings for every movie in the dataset.

This runs once and saves the result to data/embeddings.npy.
Downstream modules (recommender, Streamlit app) load these precomputed
embeddings — they're way too slow to recompute on every request.

Usage:
    python src/embedder.py
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# A small but solid model — 384-dim vectors, ~80MB download, fast on CPU.
# This is the go-to choice when you want quality embeddings without a GPU.
MODEL_NAME = "all-MiniLM-L6-v2"


def main() -> None:
    print(f"📂 Loading cleaned dataset...")
    df = pd.read_pickle(DATA_DIR / "movies_clean.pkl")
    print(f"   {len(df)} movies to embed")

    print(f"🤖 Loading model: {MODEL_NAME}")
    print("   (first run downloads ~80MB — be patient)")
    model = SentenceTransformer(MODEL_NAME)

    print(f"🧠 Computing embeddings for all movies...")
    start = time.time()
    embeddings = model.encode(
        df["soup"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # normalize so cosine sim = dot product
    )
    elapsed = time.time() - start
    print(f"   Done in {elapsed:.1f}s")
    print(f"   Shape: {embeddings.shape}  (movies × dimensions)")

    # Save embeddings
    np.save(DATA_DIR / "embeddings.npy", embeddings)
    print(f"💾 Saved to data/embeddings.npy")

    # Quick demo — what's most similar to the first movie?
    print(f"\n🔍 Demo: top 5 most similar movies to '{df.iloc[0]['title']}'")
    query_vec = embeddings[0]
    similarities = embeddings @ query_vec   # dot product with all
    top_indices = similarities.argsort()[::-1][1:6]  # skip self (index 0)
    for idx in top_indices:
        sim = similarities[idx]
        print(f"   {sim:.3f}  {df.iloc[idx]['title']}")


if __name__ == "__main__":
    main()
