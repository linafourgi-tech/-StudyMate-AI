"""Environment-based configuration for the StudyMate AI LLM backend.

Settings come from real environment variables and, optionally, from the
project's ``.env`` file (git-ignored). The ``.env`` file is located by an
explicit path relative to this source file (``<project root>/.env``), so it
is found regardless of the current working directory, REPL, or debugger.
Real environment variables take priority over values in ``.env``, and
loading ``.env`` does not modify ``os.environ``.

The API key is stored as a ``SecretStr`` so it is not printed in logs, reprs,
or configuration error messages.

Environment variables
---------------------
LLM_PROVIDER              ``openai`` (default) or ``gemini``.
OPENAI_API_KEY            API key, required when LLM_PROVIDER=openai.
GEMINI_API_KEY            API key, required when LLM_PROVIDER=gemini.
LLM_MODEL                 Model name. Default: ``gpt-4o-mini`` (openai) /
                          ``gemini-3.5-flash-lite`` (gemini).
GEMINI_THINKING_LEVEL     Optional (Gemini only): minimal, low, medium or high.
                          Unset = the model's own default.
LLM_BASE_URL              Optional OpenAI-compatible endpoint URL.
LLM_TIMEOUT_SECONDS       Timeout for one HTTP attempt. Default: 30.
LLM_TOTAL_TIMEOUT_SECONDS Time budget for one call, including all retries. Default: 60.
LLM_MAX_RETRIES           Retries for transient errors. Default: 3.
LLM_RETRY_BASE_DELAY      Base backoff delay in seconds. Default: 1.0.
LLM_RETRY_MAX_DELAY       Maximum backoff delay in seconds. Default: 20.0.
LLM_TEMPERATURE           Default sampling temperature. Default: 0.2.
LLM_MAX_OUTPUT_TOKENS     Default output token limit. Default: 1024.
LLM_VALIDATION_RETRIES    Extra attempts when structured output is invalid. Default: 1.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from dotenv import dotenv_values
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)

#: Project root (the directory that contains ``src/``).
PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: The ``.env`` file read by ``load_settings`` when no ``env_file`` is given.
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"

Provider = Literal["openai", "gemini"]

#: Default model per provider (overridden by LLM_MODEL).
DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    # Verified live on 2026-11-17 (student's Free Tier project, API v1). It is the
    # model Google names as the replacement for the deprecated gemini-2.5-flash,
    # which returned HTTP 404 for generation on this project although it was listed.
    "gemini": "gemini-3.5-flash-lite",
}

#: Environment variable holding the API key for each provider.
API_KEY_ENV: dict[str, str] = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}


class ConfigurationError(Exception):
    """Raised when required configuration is missing or invalid."""


class Settings(BaseModel):
    """Validated, immutable runtime settings for the LLM service."""

    model_config = ConfigDict(frozen=True)

    provider: Provider = "openai"
    api_key: SecretStr
    model: str = DEFAULT_MODELS["openai"]
    base_url: str | None = None
    timeout_seconds: float = Field(default=30.0, gt=0, le=600)
    total_timeout_seconds: float = Field(default=60.0, gt=0, le=1800)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_base_delay: float = Field(default=1.0, ge=0, le=60)
    retry_max_delay: float = Field(default=20.0, ge=0, le=300)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=1024, ge=1, le=32000)
    validation_retries: int = Field(default=1, ge=0, le=3)
    gemini_thinking_level: Literal["minimal", "low", "medium", "high"] | None = None

    @model_validator(mode="before")
    @classmethod
    def _default_model_for_provider(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("model"):
            provider = data.get("provider", "openai")
            if provider in DEFAULT_MODELS:
                data = {**data, "model": DEFAULT_MODELS[provider]}
        return data

    @field_validator("api_key")
    @classmethod
    def _api_key_not_blank(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("API key must not be empty")
        return value

    @field_validator("model")
    @classmethod
    def _model_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("model must not be empty")
        return value

    @model_validator(mode="after")
    def _check_retry_delays(self) -> Settings:
        if self.retry_max_delay < self.retry_base_delay:
            raise ValueError("retry_max_delay must be greater than or equal to retry_base_delay")
        return self


# Maps environment variable names to Settings field names.
# The API key variable is chosen per provider (see API_KEY_ENV).
_ENV_MAP: dict[str, str] = {
    "LLM_PROVIDER": "provider",
    "LLM_MODEL": "model",
    "LLM_BASE_URL": "base_url",
    "LLM_TIMEOUT_SECONDS": "timeout_seconds",
    "LLM_TOTAL_TIMEOUT_SECONDS": "total_timeout_seconds",
    "LLM_MAX_RETRIES": "max_retries",
    "LLM_RETRY_BASE_DELAY": "retry_base_delay",
    "LLM_RETRY_MAX_DELAY": "retry_max_delay",
    "LLM_TEMPERATURE": "temperature",
    "LLM_MAX_OUTPUT_TOKENS": "max_output_tokens",
    "LLM_VALIDATION_RETRIES": "validation_retries",
    "GEMINI_THINKING_LEVEL": "gemini_thinking_level",
}


def _read_env_file(path: Path) -> dict[str, str]:
    """Read ``KEY=value`` pairs from ``path`` without touching ``os.environ``."""
    if not path.is_file():
        return {}
    return {key: value for key, value in dotenv_values(path).items() if value is not None}


def load_settings(
    env: Mapping[str, str] | None = None,
    *,
    env_file: str | os.PathLike[str] | None = None,
    load_env_file: bool = True,
) -> Settings:
    """Build ``Settings`` from the environment.

    Args:
        env: Mapping to read from instead of the real environment (used by
            tests). When given, no ``.env`` file is read.
        env_file: Path of the ``.env`` file. Defaults to ``DEFAULT_ENV_FILE``
            (``<project root>/.env``), independent of the working directory.
        load_env_file: Set to ``False`` to ignore any ``.env`` file.

    Raises:
        ConfigurationError: if the API key is missing or any value is invalid.
            The message names the invalid fields but never contains values.
    """
    if env is None:
        merged: dict[str, str] = {}
        if load_env_file:
            merged.update(_read_env_file(Path(env_file) if env_file else DEFAULT_ENV_FILE))
        merged.update(os.environ)  # real environment variables take priority
        env = merged

    values: dict[str, Any] = {
        field: env[name].strip() for name, field in _ENV_MAP.items() if env.get(name, "").strip()
    }
    if "provider" in values:
        values["provider"] = values["provider"].lower()
    if "gemini_thinking_level" in values:
        values["gemini_thinking_level"] = values["gemini_thinking_level"].lower()

    provider = values.get("provider", "openai")
    if provider not in API_KEY_ENV:
        raise ConfigurationError(
            f"Invalid LLM configuration: provider: must be one of {sorted(API_KEY_ENV)}"
        )
    key_name = API_KEY_ENV[provider]
    if env.get(key_name, "").strip():
        values["api_key"] = env[key_name]
    else:
        raise ConfigurationError(
            f"{key_name} is not set (LLM_PROVIDER={provider}). Copy .env.example to .env in "
            "the project root and add your key (the .env file is git-ignored and must never "
            "be committed)."
        )

    try:
        return Settings(**values)
    except ValidationError as exc:
        # Report field names and reasons only; never echo input values (may contain secrets).
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'settings'}: {err['msg']}"
            for err in exc.errors()
        )
        raise ConfigurationError(f"Invalid LLM configuration: {problems}") from None
