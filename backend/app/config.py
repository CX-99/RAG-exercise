"""Typed runtime settings from environment variables and optional .env files."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    """Environment variables take precedence over optional local dotenv files.

    Tests pass _env_file=None to avoid reading local credentials.
    """

    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "openai/gpt-oss-20b"
    embedding_model: str = "openai/text-embedding-3-small"
    corpus_dir: Path = Path("data/corpus")
    retrieval_top_k: int = Field(default=5, ge=1)
    # Covers the model's hidden reasoning as well as the answer; a whole-policy summary
    # needs more than 500.
    llm_max_tokens: int = Field(default=2000, ge=1)
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    log_level: str = "INFO"

    @model_validator(mode="after")
    def require_api_key(self) -> "Settings":
        """Fail at startup, not on the first request, if there is no usable key.

        Both embeddings and generation are hosted, so nothing works without one.
        """
        key = self.openrouter_api_key
        if key is None or not key.get_secret_value().strip():
            raise ValueError("OPENROUTER_API_KEY is required: set it in .env or the environment")
        return self

    @field_validator("corpus_dir")
    @classmethod
    def resolve_corpus_dir(cls, value: Path) -> Path:
        """Resolve relative paths against the repo root, whatever the working directory."""
        return (REPO_ROOT / value).resolve()


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, read once and then cached."""
    return Settings()
