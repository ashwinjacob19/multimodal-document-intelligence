from abc import ABC, abstractmethod

from app.reranking.models import RerankMetadata
from app.retrieval.models import RetrievedChunk


class RerankProvider(ABC):
    @property
    @abstractmethod
    def metadata(self) -> RerankMetadata:
        """Returns metadata configuration details for this reranker provider."""
        pass

    @abstractmethod
    def rerank(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Reranks retrieved chunks for the query in descending relevance."""
        pass


class CrossEncoderRerankProvider(RerankProvider):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    @property
    def metadata(self) -> RerankMetadata:
        return RerankMetadata(provider="sbert", model_name=self.model_name)

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Reranks chunks locally using the loaded CrossEncoder model."""
        if not chunks:
            return []

        pairs = [[query, chunk.text] for chunk in chunks]
        scores = self.model.predict(pairs)

        reranked = [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                similarity=chunk.similarity,
                rerank_score=float(score),
            )
            for chunk, score in zip(chunks, scores, strict=True)
        ]

        # Return sorted list using deterministic ordering
        return sorted(
            reranked,
            key=lambda c: (
                -(
                    c.rerank_score
                    if c.rerank_score is not None
                    else float("-inf")
                ),
                c.chunk_index,
            ),
        )


class FakeRerankProvider(RerankProvider):
    @property
    def metadata(self) -> RerankMetadata:
        return RerankMetadata(provider="fake", model_name="mock-model")

    def rerank(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Returns deterministically reversed order of chunks using mock scores."""
        reranked = [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                similarity=chunk.similarity,
                rerank_score=float(i + 1),
            )
            for i, chunk in enumerate(chunks)
        ]

        # Return sorted list using deterministic ordering
        return sorted(
            reranked,
            key=lambda c: (
                -(
                    c.rerank_score
                    if c.rerank_score is not None
                    else float("-inf")
                ),
                c.chunk_index,
            ),
        )
