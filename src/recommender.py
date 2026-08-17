"""
The RAG brain of the movie recommender (Powered by Local Ollama).

Combines:
  1. Semantic search over precomputed movie embeddings (retrieval)
  2. Local Ollama (llama3.2) to generate a natural-language response with
     real movie recommendations from the dataset (generation)

This is what gets called from the Streamlit chat UI.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import ollama
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2"  # Fast, local, and completely free


SYSTEM_PROMPT = """You are LBH Cima, a friendly movie recommendation assistant.

You help users find movies based on their mood, recent watches, or specific tastes.
You ONLY recommend movies from the candidate list provided to you for each query —
never invent titles. If none of the candidates fit the user's request, say so
honestly and ask a clarifying question instead of forcing a bad match.

Style:
- Conversational and warm, not robotic
- Recommend 3-5 movies, never more
- For each recommendation, give a one-sentence reason tied to what the user asked
- Format each recommendation like: **Title (Year)** — your reason

When the user is just chatting (no recommendation request), respond naturally
and helpfully without forcing recommendations."""


class MovieRecommender:
    """Loads the dataset, embeddings, and models once. Reused across queries."""

    def __init__(self) -> None:
        print("Loading movie dataset...")
        self.df = pd.read_pickle(DATA_DIR / "movies_clean.pkl")

        print("Loading precomputed embeddings...")
        self.embeddings = np.load(DATA_DIR / "embeddings.npy")

        print("Loading sentence-transformer model...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        print(f"Connecting to local Ollama model ({OLLAMA_MODEL})...")
        print("Ready.")

    def retrieve(self, query: str, top_k: int = 10) -> pd.DataFrame:
        """Find the top_k movies most semantically similar to the query."""
        query_vec = self.embedder.encode(
            query, normalize_embeddings=True, convert_to_numpy=True
        )
        similarities = self.embeddings @ query_vec
        top_indices = similarities.argsort()[::-1][:top_k]
        results = self.df.iloc[top_indices].copy()
        results["similarity"] = similarities[top_indices]
        return results

    def format_candidates(self, candidates: pd.DataFrame) -> str:
        """Turn the retrieved DataFrame into a string the LLM can read."""
        lines = []
        for _, row in candidates.iterrows():
            year = (
                pd.to_datetime(row["release_date"], errors="coerce").year
                if pd.notna(row["release_date"]) else "Unknown"
            )
            genres = ", ".join(row["genres"]) if row["genres"] else "Unknown"
            director = row["director"] or "Unknown"
            overview = (row["overview"] or "")[:200]
            lines.append(
                f"- **{row['title']} ({year})** | Genres: {genres} | "
                f"Director: {director} | Rating: {row['vote_average']}/10\n"
                f"  Plot: {overview}"
            )
        return "\n".join(lines)

    def _build_messages(self, user_message: str, history: list[dict] | None = None) -> list[dict]:
        """Format chat history and current prompt for Ollama."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if history:
            for item in history:
                # Adapt from previous Gemini format if needed
                role = "assistant" if item.get("role") in ["model", "assistant"] else "user"
                content = item.get("content") or item.get("parts", [""])[0]
                messages.append({"role": role, "content": content})

        candidates = self.retrieve(user_message, top_k=10)
        candidate_text = self.format_candidates(candidates)

        augmented_prompt = f"""User message: "{user_message}"

Here are the 10 movies from our database that are most semantically relevant
to what the user just said. Recommend the best 3-5 of these for the user
(or fewer if only a few really fit). Do NOT recommend any movie outside this list.

CANDIDATES:
{candidate_text}"""

        messages.append({"role": "user", "content": augmented_prompt})
        return messages

    def chat(self, user_message: str, history: list[dict] | None = None) -> str:
        """Main entry point — RAG using local Ollama."""
        messages = self._build_messages(user_message, history)
        response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
        return response["message"]["content"]

    def chat_stream(self, user_message: str, history: list[dict] | None = None):
        """Streaming version using local Ollama for the typewriter effect."""
        messages = self._build_messages(user_message, history)
        try:
            stream = ollama.chat(model=OLLAMA_MODEL, messages=messages, stream=True)
            for chunk in stream:
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
        except Exception as e:
            yield f"⚠️ Ollama error: {str(e)}. Make sure the Ollama application is running on your Mac!"


if __name__ == "__main__":
    rec = MovieRecommender()
    print("\n" + "=" * 60)
    print("LBH Cima CLI test (Ollama) — type 'quit' to exit")
    print("=" * 60)
    history = []
    while True:
        msg = input("\nYou: ").strip()
        if msg.lower() in {"quit", "exit", "q"}:
            break
        if not msg:
            continue
        reply = rec.chat(msg, history)
        print(f"\nLBH Cima: {reply}")
        history.append({"role": "user", "content": msg})
        history.append({"role": "assistant", "content": reply})