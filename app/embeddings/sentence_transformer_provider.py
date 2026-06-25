import logging

from sentence_transformers import SentenceTransformer

from app.config import settings
from app.embeddings.models import EmbeddingMetadata, EmbeddingResult
from app.embeddings.provider import EmbeddingProvider

logger = logging.getLogger(__name__)


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    _model: SentenceTransformer | None = None
    _metadata: EmbeddingMetadata | None = None

    @classmethod
    def _load_model(cls) -> None:
        """Loads and caches the SentenceTransformer model and metadata class-wide."""
        if cls._model is None:
            model_name = settings.EMBEDDING_MODEL
            cls._model = SentenceTransformer(model_name)
            dimension = cls._model.get_embedding_dimension()
            cls._metadata = EmbeddingMetadata(
                model_name=model_name,
                dimension=dimension,
            )
            # Log loading message matching target output format
            msg = f"Embedding model loaded\n\nModel:\n{model_name}\n"
            print(msg, flush=True)
            logger.info("Embedding model loaded. Model: %s", model_name)

    @property
    def metadata(self) -> EmbeddingMetadata:
        """Exposes the model metadata dynamically."""
        self._load_model()
        assert self._metadata is not None
        return self._metadata

    def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generates embeddings for the provided list of texts."""
        if not texts:
            return []

        self._load_model()
        assert self._model is not None
        assert self._metadata is not None

        # Generate normalized embeddings optimized for cosine similarity
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        dim = self._metadata.dimension

        # Print structured logging to match target format
        progress_msg = (
            "Generating embeddings...\n\n"
            f"Batch Size:\n{len(texts)}\n\n"
            f"Embedding Dimension:\n{dim}\n\n"
            "Completed successfully.\n"
        )
        print(progress_msg, flush=True)
        logger.info(
            "Generating embeddings. Batch Size: %d, "
            "Dimension: %d. Completed successfully.",
            len(texts),
            dim,
        )

        results = []
        for text, emb in zip(texts, embeddings, strict=False):
            results.append(
                EmbeddingResult(
                    text=text,
                    embedding=emb.tolist(),  # Never expose numpy arrays
                )
            )
        return results
