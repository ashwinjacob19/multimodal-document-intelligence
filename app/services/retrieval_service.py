from sqlalchemy.ext.asyncio import AsyncSession

from app.retrieval.models import RetrievalContext, RetrievalMetadata, RetrievedChunk
from app.search.models import SearchQuery
from app.services.search_service import SearchService


class RetrievalService:
    def __init__(self, search_service: SearchService):
        """Initializes RetrievalService with an injected SearchService dependency."""
        self.search_service = search_service

    async def retrieve(
        self, db: AsyncSession, query: SearchQuery
    ) -> RetrievalContext:
        """Retrieves semantically relevant document chunks, preserving search order,

        and packages them into a structured RetrievalContext.
        """
        # 1. Fetch chunks using the injected SearchService
        search_results = await self.search_service.search(db, query)

        # 2. Map search results to RetrievedChunk dataclasses (maintaining ordering)
        retrieved_chunks = [
            RetrievedChunk(
                chunk_id=res.chunk_id,
                document_id=res.document_id,
                page_number=res.page_number,
                chunk_index=res.chunk_index,
                text=res.text,
                similarity=res.similarity,
            )
            for res in search_results
        ]

        # 3. Compile metadata block
        metadata = RetrievalMetadata(
            query=query.text,
            retrieved_count=len(retrieved_chunks),
            document_id=query.document_id,
            limit=query.limit,
        )

        # 4. Return RetrievalContext package
        return RetrievalContext(
            chunks=retrieved_chunks,
            metadata=metadata,
        )
