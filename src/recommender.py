"""
The RAG brain of the movie recommender.

Combines:
  1. Semantic search over precomputed movie embeddings (retrieval)
  2. Google Gemini to generate a natural-language response with
     real movie recommendations from the dataset (generation)

This is what gets called from the Streamlit chat UI.
"""

import os
import numpy as np
import pandas as pd
import google.generativeai as genai
from pathlib import Path
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash-lite"  # fast + free tier friendly

# Load environment variables from .env
load_dotenv(ROOT / ".env")


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

        print("Configuring Gemini...")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not found. "
                "Make sure .env exists in the project root and contains your key."
            )
        genai.configure(api_key=api_key)
        self.llm = genai.GenerativeModel(
            GEMINI_MODEL,
            system_instruction=SYSTEM_PROMPT,
        )
        print("Ready.")

    def retrieve(self, query: str, top_k: int = 10) -> pd.DataFrame:
        """Find the top_k movies most semantically similar to the query."""
        query_vec = self.embedder.encode(
            query, normalize_embeddings=True, convert_to_numpy=True
        )
        # Embeddings already normalized → dot product = cosine similarity
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
            overview = (row["overview"] or "")[:200]  # truncate for token budget
            lines.append(
                f"- **{row['title']} ({year})** | Genres: {genres} | "
                f"Director: {director} | Rating: {row['vote_average']}/10\n"
                f"  Plot: {overview}"
            )
        return "\n".join(lines)

    def chat(self, user_message: str, history: list[dict] | None = None) -> str:
        """Main entry point — RAG: retrieve relevant movies, then generate."""
        history = history or []

        # 1. RETRIEVAL — get top candidates from our dataset
        candidates = self.retrieve(user_message, top_k=10)
        candidate_text = self.format_candidates(candidates)

        # 2. AUGMENTATION — build a prompt with the user's message + candidates
        augmented_prompt = f"""User message: "{user_message}"

Here are the 10 movies from our database that are most semantically relevant
to what the user just said. Recommend the best 3-5 of these for the user
(or fewer if only a few really fit). Do NOT recommend any movie outside this list.

CANDIDATES:
{candidate_text}"""

        # 3. GENERATION — Gemini composes the natural-language response
        # We pass the prior conversation so it has chat memory
        chat = self.llm.start_chat(history=history)
        response = chat.send_message(augmented_prompt)
        return response.text



    def chat_stream(self, user_message: str, history: list[dict] | None = None):
        """Streaming version — yields chunks as Gemini generates them.

        Used by the Streamlit UI for a typewriter-style response that
        appears progressively rather than all at once.
        """
        history = history or []

        # 1. RETRIEVAL — same as chat()
        candidates = self.retrieve(user_message, top_k=10)
        candidate_text = self.format_candidates(candidates)

        # 2. AUGMENTATION
        augmented_prompt = f"""User message: "{user_message}"

Here are the 10 movies from our database that are most semantically relevant
to what the user just said. Recommend the best 3-5 of these for the user
(or fewer if only a few really fit). Do NOT recommend any movie outside this list.

CANDIDATES:
{candidate_text}"""

        # 3. STREAMING GENERATION
        chat = self.llm.start_chat(history=history)
        try:
            response = chat.send_message(augmented_prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                yield (
                    "⏳ I'm getting a lot of requests right now and hit my "
                    "free-tier limit. Please try again in a minute, or come "
                    "back tomorrow — it resets daily. Thanks for your patience!"
                )
            else:
                raise


if __name__ == "__main__":
    # Quick CLI test — run `python src/recommender.py` to verify it works
    rec = MovieRecommender()
    print("\n" + "=" * 60)
    print("LBH Cima CLI test — type 'quit' to exit")
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
        # Track history so context carries forward
        history.append({"role": "user", "parts": [msg]})
        history.append({"role": "model", "parts": [reply]})
