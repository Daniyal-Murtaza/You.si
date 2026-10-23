"""Runtime settings, from the environment or ``.env``."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    provider: Literal["openai", "ollama", "fake"] = Field(default="openai", alias="YOUSI_PROVIDER")
    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")
    ollama_model: str = Field(default="llama3.1:8b", alias="OLLAMA_MODEL")
    ollama_url: str = Field(default="http://localhost:11434", alias="OLLAMA_URL")
    model_timeout_s: float = Field(default=90.0, alias="YOUSI_MODEL_TIMEOUT_S")

    vault: Path = Field(default=Path("./vaults/me"), alias="YOUSI_VAULT")
    embed_model: str = Field(default="BAAI/bge-small-en-v1.5", alias="YOUSI_EMBED_MODEL")
    log_level: str = Field(default="INFO", alias="YOUSI_LOG_LEVEL")

    # Budgets, enforced in code.
    max_questions_per_session: int = Field(default=8, ge=1, alias="YOUSI_MAX_QUESTIONS")
    retrieval_top_k: int = Field(default=8, ge=1, le=30, alias="YOUSI_TOP_K")
    chunk_target_words: int = Field(default=160, alias="YOUSI_CHUNK_WORDS")

    def require_openai(self) -> str:
        if self.openai_api_key is None:
            raise MissingCredentialError("OPENAI_API_KEY")
        return self.openai_api_key.get_secret_value()


class MissingCredentialError(RuntimeError):
    def __init__(self, name: str) -> None:
        super().__init__(f"{name} is not set. Copy .env.example to .env and fill it in.")
        self.name = name


def load_settings(**overrides: object) -> Settings:
    return Settings(**overrides)  # type: ignore[arg-type]
