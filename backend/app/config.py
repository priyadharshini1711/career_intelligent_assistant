"""Application configuration.

Everything tunable lives here so that behaviour can be changed through the
environment without touching code. Defaults are chosen so that `uvicorn
app.main:app` works out of the box with no `.env` file at all -- in that case
the deterministic stub LLM is used, which keeps tests and first-run experience
fast and offline.
"""

from functools import lru_cache
from typing import List, Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProviderName = Literal["gemini", "groq", "ollama", "stub"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Service ---------------------------------------------------------
    app_name: str = "Career Intelligence Assistant"
    environment: Literal["local", "staging", "production"] = "local"
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "console"
    cors_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )

    # --- LLM -------------------------------------------------------------
    # `stub` is the default so the app boots and the test suite runs without
    # any API key. Set LLM_PROVIDER=gemini plus GEMINI_API_KEY for real answers.
    llm_provider: LLMProviderName = "stub"
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2
    llm_temperature: float = 0.2
    llm_max_output_tokens: int = 1200

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash"

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    # --- Embeddings ------------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 32

    # --- Chunking --------------------------------------------------------
    # Sized in tokens-ish (words). Resumes and JDs are short documents, so
    # small chunks with generous overlap retrieve better than large ones.
    chunk_target_words: int = 120
    chunk_overlap_words: int = 30
    chunk_min_words: int = 15

    # --- Retrieval -------------------------------------------------------
    retrieval_top_k: int = 6
    retrieval_candidate_k: int = 20
    # Weight of dense (vector) score when fused with lexical BM25 score.
    retrieval_dense_weight: float = 0.65
    retrieval_mmr_lambda: float = 0.6
    # Leave unset to use the embedding model's own weak-evidence threshold.
    # Set it only to override for a specific deployment.
    retrieval_min_score: Optional[float] = None

    # --- Context budget --------------------------------------------------
    # Rough word budget for retrieved context. Kept well below the model
    # window so the instructions and question are never truncated away.
    context_max_words: int = 1800

    # --- Uploads ---------------------------------------------------------
    max_upload_bytes: int = 5 * 1024 * 1024
    max_jobs_per_session: int = 10
    allowed_extensions: List[str] = Field(
        default_factory=lambda: [".pdf", ".docx", ".txt", ".md"]
    )

    # Bundled demo documents. Defaults to `<repo>/samples`, which is also where
    # the compose file mounts them in the container -- but relying on a
    # relative path resolving the same way in both layouts is the kind of
    # coincidence that breaks silently, so it is configurable.
    samples_dir: str = ""

    # --- Sessions --------------------------------------------------------
    session_ttl_seconds: int = 60 * 60 * 4
    max_sessions: int = 200
    max_chat_history_turns: int = 6


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor so settings are parsed once per process."""
    return Settings()
