import asyncio

import httpx
import pytest

from app.db.models.chunk import Chunk
from app.db.models.document import Document, DocumentStatus, DocumentType
from app.db.session import AsyncSessionLocal
from app.embeddings.factory import EmbeddingProviderFactory
from app.main import app
from app.search.models import SearchQuery
from app.services.retrieval_service import RetrievalService
from app.services.search_service import SearchService


def test_retrieval_dependency_injection() -> None:
    """Verifies RetrievalService dependency injection."""
    search_provider = EmbeddingProviderFactory.create()
    search_service = SearchService(search_provider)
    retrieval_service = RetrievalService(search_service)

    assert retrieval_service.search_service is search_service


@pytest.mark.asyncio
async def test_retrieval_context_and_order_preservation() -> None:
    """Verifies that RetrievalService preserves SearchService ranking."""
    provider = EmbeddingProviderFactory.create()
    search_service = SearchService(provider)
    retrieval_service = RetrievalService(search_service)

    async with AsyncSessionLocal() as db:
        # Create test doc and chunks
        doc = Document(
            filename="retrieval_test.pdf",
            document_type=DocumentType.pdf,
            status=DocumentStatus.uploaded,
            source="test.pdf",
        )
        db.add(doc)
        await db.commit()
        await db.refresh(doc)

        text1 = "Apples are red fruit."
        text2 = "Bananas are yellow fruit."

        embedding_results = await asyncio.to_thread(provider.embed, [text1, text2])
        emb1 = embedding_results[0].embedding
        emb2 = embedding_results[1].embedding

        chunk1 = Chunk(
            document_id=doc.id,
            page_number=1,
            chunk_index=0,
            text=text1,
            char_count=len(text1),
            embedding=emb1,
        )
        chunk2 = Chunk(
            document_id=doc.id,
            page_number=1,
            chunk_index=1,
            text=text2,
            char_count=len(text2),
            embedding=emb2,
        )
        db.add_all([chunk1, chunk2])
        await db.commit()

        try:
            # 1. Search directly
            query = SearchQuery(text="red fruit", limit=5)
            search_results = await search_service.search(db, query)

            # 2. Retrieve via RetrievalService
            context = await retrieval_service.retrieve(db, query)

            # Assert order is preserved exactly
            assert len(context.chunks) == len(search_results)
            for ctx_chunk, s_chunk in zip(context.chunks, search_results, strict=True):
                assert ctx_chunk.chunk_id == s_chunk.chunk_id
                assert ctx_chunk.text == s_chunk.text
                assert ctx_chunk.similarity == s_chunk.similarity

            # Verify metadata
            assert context.metadata.query == "red fruit"
            assert context.metadata.retrieved_count == len(search_results)
            assert context.metadata.limit == 5
            assert context.metadata.document_id is None

            # Test empty retrieval
            empty_query = SearchQuery(text="", limit=5)
            empty_context = await retrieval_service.retrieve(db, empty_query)
            assert len(empty_context.chunks) == 0
            assert empty_context.metadata.retrieved_count == 0

        finally:
            await db.delete(chunk1)
            await db.delete(chunk2)
            await db.delete(doc)
            await db.commit()


@pytest.mark.asyncio
async def test_retrieval_api_limit_clamping_and_regression() -> None:
    """Verifies API serialization, limit clamping, and rank regression."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as ac:
        # 1. Verify limit clamping to MAX_SEARCH_LIMIT (20)
        response_clamp = await ac.get(
            "/retrieval",
            params={"q": "apples", "limit": 100},
        )
        assert response_clamp.status_code == 200
        res_data = response_clamp.json()
        assert "chunks" in res_data
        assert "metadata" in res_data
        assert res_data["metadata"]["limit"] == 20  # Clamped

        # 2. Verify exact rank regression with GET /search/chunks
        query_text = "yellow fruit"
        response_search = await ac.get(
            "/search/chunks",
            params={"q": query_text, "limit": 5},
        )
        response_retrieval = await ac.get(
            "/retrieval",
            params={"q": query_text, "limit": 5},
        )

        assert response_search.status_code == 200
        assert response_retrieval.status_code == 200

        search_chunks = response_search.json()
        retrieval_chunks = response_retrieval.json()["chunks"]

        assert len(search_chunks) == len(retrieval_chunks)
        for s_chunk, r_chunk in zip(search_chunks, retrieval_chunks, strict=True):
            assert s_chunk["chunk_id"] == r_chunk["chunk_id"]
            assert s_chunk["text"] == r_chunk["text"]
            assert s_chunk["similarity"] == pytest.approx(r_chunk["similarity"])
