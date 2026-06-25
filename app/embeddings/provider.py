from abc import ABC, abstractmethod

from app.embeddings.models import EmbeddingMetadata, EmbeddingResult


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def metadata(self) -> EmbeddingMetadata:
        """Returns metadata about the embedding model/provider."""
        pass

    @abstractmethod
    def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generates embeddings for a list of texts (synchronous)."""
        pass
