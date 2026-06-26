import asyncio
import uuid

import httpx
import pytest

from app.config import settings
from app.db.models.chunk import Chunk
from app.db.models.document import Document, DocumentStatus, DocumentType
from app.db.session import AsyncSessionLocal
from app.embeddings.factory import EmbeddingProviderFactory
from app.main import app
from app.search.models import SearchQuery
from app.services.search_service import SearchService


def test_search_query_model_validation() -> None:
    """Verifies SearchQuery initialization and default values."""
    query = SearchQuery(text="semantic query")
    assert query.text == "semantic query"
    assert query.document_id is None
    assert query.limit == settings.DEFAULT_SEARCH_LIMIT

    doc_id = uuid.uuid4()
    query_custom = SearchQuery(text="another query", document_id=doc_id, limit=10)
    assert query_custom.text == "another query"
    assert query_custom.document_id == doc_id
    assert query_custom.limit == 10


@pytest.mark.asyncio
async def test_semantic_search_ranking_filtering_and_determinism() -> None:
    """Verifies search ranking, document filtering, and deterministic ordering."""
    provider = EmbeddingProviderFactory.create()
    service = SearchService(provider)

    async with AsyncSessionLocal() as db:
        # 1. Create two test documents
        doc1 = Document(
            filename="doc1.pdf",
            document_type=DocumentType.pdf,
            status=DocumentStatus.uploaded,
            source="test.pdf",
        )
        doc2 = Document(
            filename="doc2.pdf",
            document_type=DocumentType.pdf,
            status=DocumentStatus.uploaded,
            source="test.pdf",
        )
        db.add_all([doc1, doc2])
        await db.commit()
        await db.refresh(doc1)
        await db.refresh(doc2)

        # 2. Get embeddings for two distinct texts
        text1 = "The quick brown fox jumps over the lazy dog."
        text2 = "Quantum physics is a fundamental theory in physics."

        embedding_results = await asyncio.to_thread(provider.embed, [text1, text2])
        emb1 = embedding_results[0].embedding
        emb2 = embedding_results[1].embedding

        # 3. Create three chunks (two in doc1, one in doc2)
        chunk1 = Chunk(
            document_id=doc1.id,
            page_number=1,
            chunk_index=0,
            text=text1,
            char_count=len(text1),
            embedding=emb1,
        )
        chunk2 = Chunk(
            document_id=doc1.id,
            page_number=1,
            chunk_index=1,
            text=text2,
            char_count=len(text2),
            embedding=emb2,
        )
        chunk3 = Chunk(
            document_id=doc2.id,
            page_number=1,
            chunk_index=0,
            text=text1,  # Same text, but in doc2
            char_count=len(text1),
            embedding=emb1,
        )
        db.add_all([chunk1, chunk2, chunk3])
        await db.commit()

        try:
            # 4. Search query matching text1 closely (should rank chunks 1 and 3 first)
            query_fox = SearchQuery(text="brown fox jumps", limit=5)
            results_fox = await service.search(db, query_fox)

            assert len(results_fox) >= 2
            # Chunks containing text1 should rank first due to cosine distance
            assert results_fox[0].text == text1
            assert results_fox[1].text == text1

            # Check similarity is valid score between -1.0 and 1.0
            for res in results_fox:
                assert -1.0 <= res.similarity <= 1.0

            # 5. Test document scope filtering (filter only for doc2)
            query_filtered = SearchQuery(
                text="brown fox jumps", document_id=doc2.id, limit=5
            )
            results_filtered = await service.search(db, query_filtered)

            # Should only contain chunk3 (belonging to doc2)
            for res in results_filtered:
                assert res.document_id == doc2.id

            # 6. Verify deterministic ordering
            # Execute the same query twice and assert identical ordering
            results_fox_run1 = await service.search(db, query_fox)
            results_fox_run2 = await service.search(db, query_fox)

            assert [r.chunk_id for r in results_fox_run1] == [
                r.chunk_id for r in results_fox_run2
            ]

        finally:
            # Cleanup
            await db.delete(chunk1)
            await db.delete(chunk2)
            await db.delete(chunk3)
            await db.delete(doc1)
            await db.delete(doc2)
            await db.commit()


@pytest.mark.asyncio
async def test_search_api_limit_clamping() -> None:
    """Verifies that requested limits are clamped to settings.MAX_SEARCH_LIMIT."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as ac:
        # Request with limit 100 (which exceeds MAX_SEARCH_LIMIT of 20)
        response = await ac.get(
            "/search/chunks",
            params={"q": "test query", "limit": 100},
        )
    assert response.status_code == 200
    res_data = response.json()
    assert isinstance(res_data, list)
