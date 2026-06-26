import asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.chunk import Chunk
from app.embeddings.provider import EmbeddingProvider
from app.search.models import SearchQuery, SearchResult


class SearchService:
    def __init__(self, embedding_provider: EmbeddingProvider):
        """Initializes SearchService with an injected EmbeddingProvider."""
        self.embedding_provider = embedding_provider

    async def search(self, db: AsyncSession, query: SearchQuery) -> list[SearchResult]:
        """Performs semantic vector similarity search over chunks table using pgvector."""
        if not query.text.strip():
            return []

        # 1. Generate query embedding (CPU-bound local sentence-transformers, offloaded to thread pool)
        query_results = await asyncio.to_thread(
            self.embedding_provider.embed, [query.text]
        )
        if not query_results:
            return []
        query_vector = query_results[0].embedding

        # 2. Build pgvector similarity query
        # Cosine distance represents 1 - cosine similarity
        distance = Chunk.embedding.cosine_distance(query_vector)
        similarity = 1.0 - distance

        stmt = select(Chunk, similarity.label("similarity")).where(
            Chunk.embedding.isnot(None)
        )

        # Apply document filter if requested
        if query.document_id is not None:
            stmt = stmt.where(Chunk.document_id == query.document_id)

        # Apply deterministic ordering (by distance first, then by chunk_index)
        stmt = stmt.order_by(distance, Chunk.chunk_index).limit(query.limit)

        res = await db.execute(stmt)
        rows = res.all()

        # 3. Map to SearchResult application dataclasses
        return [
            SearchResult(
                chunk_id=row.Chunk.id,
                document_id=row.Chunk.document_id,
                page_number=row.Chunk.page_number,
                chunk_index=row.Chunk.chunk_index,
                text=row.Chunk.text,
                similarity=float(row.similarity),
            )
            for row in rows
        ]
