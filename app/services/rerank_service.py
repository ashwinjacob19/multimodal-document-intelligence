import asyncio

from app.reranking.provider import RerankProvider
from app.retrieval.models import RetrievedChunk


class RerankService:
    def __init__(self, provider: RerankProvider):
        """Initializes RerankService with injected dependencies."""
        self.provider = provider

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Reranks retrieved chunks, offloading local prediction to a threadpool."""
        if not chunks:
            return []
        return await asyncio.to_thread(self.provider.rerank, query, chunks)
