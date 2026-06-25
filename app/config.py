from typing import Any

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ENVIRONMENT: str = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    UPLOAD_DIR: str = "/app/uploads"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    EMBEDDING_PROVIDER: str = "sentence-transformers"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "document_intelligence"
    POSTGRES_HOST: str = "db"
    POSTGRES_PORT: int = 5432

    DATABASE_URL: str = ""

    @model_validator(mode="before")
    @classmethod
    def assemble_db_url(cls, data: Any) -> Any:
        if isinstance(data, dict):
            db_url = data.get("DATABASE_URL")
            if not db_url:
                user = data.get("POSTGRES_USER", "postgres")
                password = data.get("POSTGRES_PASSWORD", "postgres")
                host = data.get("POSTGRES_HOST", "db")
                port = data.get("POSTGRES_PORT", 5432)
                db = data.get("POSTGRES_DB", "document_intelligence")
                data["DATABASE_URL"] = (
                    f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"
                )
        return data


settings = Settings()
