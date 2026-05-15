"""
Loads and cleans the TMDB 5000 dataset.

Run this once to produce data/movies_clean.pkl, which downstream
modules (the embedder, the recommender, the Streamlit app) consume.

Usage:
    python src/data_loader.py
"""

import ast
import pandas as pd
from pathlib import Path


# Project root = parent of src/
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def parse_names(json_str: str) -> list[str]:
    """Parse a JSON-like string column into a list of 'name' values.

    TMDB stores genres/keywords/companies as stringified JSON like:
        '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]'
    We want: ['Action', 'Adventure']
    """
    if pd.isna(json_str):
        return []
    try:
        parsed = ast.literal_eval(json_str)
        return [item["name"] for item in parsed]
    except (ValueError, SyntaxError):
        return []


def parse_top_cast(json_str: str, n: int = 3) -> list[str]:
    """Top N cast members per movie is plenty for our use case."""
    if pd.isna(json_str):
        return []
    try:
        parsed = ast.literal_eval(json_str)
        return [item["name"] for item in parsed[:n]]
    except (ValueError, SyntaxError):
        return []


def parse_director(json_str: str) -> str | None:
    """Pull the director out of the crew JSON."""
    if pd.isna(json_str):
        return None
    try:
        parsed = ast.literal_eval(json_str)
        for member in parsed:
            if member.get("job") == "Director":
                return member["name"]
    except (ValueError, SyntaxError):
        return None
    return None


def load_and_clean() -> pd.DataFrame:
    """Load both CSVs, merge, parse, drop bad rows, return a clean DataFrame."""
    print("📂 Loading raw CSVs...")
    movies = pd.read_csv(DATA_DIR / "tmdb_5000_movies.csv")
    credits = pd.read_csv(DATA_DIR / "tmdb_5000_credits.csv")
    print(f"   Movies: {movies.shape}, Credits: {credits.shape}")

    # Merge cast/crew into movies
    credits = credits.rename(columns={"movie_id": "id"})
    df = movies.merge(credits[["id", "cast", "crew"]], on="id", how="left")

    print("🧹 Parsing messy JSON columns...")
    df["genres"] = df["genres"].apply(parse_names)
    df["keywords"] = df["keywords"].apply(parse_names)
    df["production_companies"] = df["production_companies"].apply(parse_names)
    df["cast"] = df["cast"].apply(parse_top_cast)
    df["director"] = df["crew"].apply(parse_director)

    # Drop movies with no overview — without one we can't recommend on content
    before = len(df)
    df = df.dropna(subset=["overview"]).reset_index(drop=True)
    print(f"   Dropped {before - len(df)} movies with no overview")

    # Build the "soup" — the text we'll embed for semantic search
    # We weight director and genres by repeating them so they matter more
    def build_soup(row) -> str:
        parts = [
            row["title"],
            row["tagline"] if pd.notna(row["tagline"]) else "",
            row["overview"],
            " ".join(row["genres"]) * 2,        # genres count double
            " ".join(row["keywords"]),
            " ".join(row["cast"]),
            (row["director"] or "") * 2,         # director counts double
        ]
        return " ".join(p for p in parts if p)

    df["soup"] = df.apply(build_soup, axis=1)

    # Keep only what we need downstream
    clean = df[[
        "id", "title", "overview", "genres", "keywords",
        "cast", "director", "vote_average", "vote_count",
        "release_date", "runtime", "tagline", "soup",
    ]].copy()

    print(f"✅ Final dataset: {len(clean)} movies")
    return clean


def main() -> None:
    df = load_and_clean()
    output = DATA_DIR / "movies_clean.pkl"
    df.to_pickle(output)
    print(f"💾 Saved to {output.relative_to(ROOT)}")

    # Quick sanity print
    print("\n📊 Sample row:")
    sample = df.iloc[0]
    print(f"   Title:    {sample['title']}")
    print(f"   Director: {sample['director']}")
    print(f"   Genres:   {sample['genres']}")
    print(f"   Soup:     {sample['soup'][:150]}...")


if __name__ == "__main__":
    main()
