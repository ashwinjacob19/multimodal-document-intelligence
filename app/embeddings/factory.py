from app.config import settings
from app.embeddings.provider import EmbeddingProvider


class EmbeddingProviderFactory:
    _instance: EmbeddingProvider | None = None

    @classmethod
    def create(cls) -> EmbeddingProvider:
        """Returns a cached singleton instance of the configured EmbeddingProvider."""
        if cls._instance is None:
            provider_type = settings.EMBEDDING_PROVIDER
            if provider_type == "sentence-transformers":
                from app.embeddings.sentence_transformer_provider import (
                    SentenceTransformerEmbeddingProvider,
                )

                cls._instance = SentenceTransformerEmbeddingProvider()
            else:
                raise ValueError(
                    f"Unsupported embedding provider: {provider_type}"
                )
        return cls._instance
