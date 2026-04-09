"""
ARGUS — Centralized Configuration Layer

Pydantic-settings based config singleton. Reads environment variables
(case-insensitive) with .env file fallback. Fails fast on missing values.

Usage:
    from src.config import settings
    print(settings.es_url)
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from enum import Enum


class Environment(str, Enum):
    development = "development"
    staging = "staging"
    production = "production"


class Settings(BaseSettings):
    """Central configuration for all ARGUS services."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Required — no defaults
    es_url: str = Field(..., description="Elasticsearch cluster URL")
    kafka_bootstrap: str = Field(..., description="Kafka bootstrap server(s)")
    redis_url: str = Field(..., description="Redis connection URL")
    mlflow_uri: str = Field(..., description="MLflow tracking URI")

    # Optional — sensible defaults
    model_name: str = Field(default="logbert_v1", description="Active model name in MLflow")
    log_level: str = Field(default="INFO", description="Logging verbosity")
    env: Environment = Field(default=Environment.development, description="Runtime environment")

    @property
    def is_production(self) -> bool:
        return self.env == Environment.production

    @property
    def is_debug(self) -> bool:
        return self.log_level == "DEBUG" or self.env == Environment.development


settings = Settings()
