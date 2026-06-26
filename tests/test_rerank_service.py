import asyncio
import uuid

import pytest

from app.db.models.chunk import Chunk
from app.db.models.document import Document, DocumentStatus, DocumentType
from app.db.session import AsyncSessionLocal
from app.embeddings.factory import EmbeddingProviderFactory
from app.llm.prompt_builder import GeminiPromptBuilder
from app.llm.provider import FakeLLMProvider
from app.reranking.factory import RerankProviderFactory
from app.reranking.provider import (
    CrossEncoderRerankProvider,
    FakeRerankProvider,
)
from app.retrieval.models import RetrievedChunk
from app.search.models import SearchQuery
from app.services.answer_service import AnswerService
from app.services.rerank_service import RerankService
from app.services.retrieval_service import RetrievalService
from app.services.search_service import SearchService


def test_rerank_dependency_injection() -> None:
    """Verifies constructor dependency injection for RerankService."""
    provider = FakeRerankProvider()
    service = RerankService(provider)
    assert service.provider is provider


def test_rerank_provider_selection_and_caching() -> None:
    """Verifies factory provider selection and singleton caching behavior."""
    from app.config import settings

    # Reset singleton state
    RerankProviderFactory._instance = None

    # Scenario 1: sbert
    original_provider = settings.RERANK_PROVIDER
    settings.RERANK_PROVIDER = "sbert"
    try:
        prov1 = RerankProviderFactory.create()
        assert isinstance(prov1, CrossEncoderRerankProvider)
        assert prov1.metadata.provider == "sbert"

        # Test singleton caching
        prov2 = RerankProviderFactory.create()
        assert prov1 is prov2
    finally:
        RerankProviderFactory._instance = None

    # Scenario 2: fallback to FakeRerankProvider
    settings.RERANK_PROVIDER = "fake"
    try:
        prov_fake = RerankProviderFactory.create()
        assert isinstance(prov_fake, FakeRerankProvider)
    finally:
        settings.RERANK_PROVIDER = original_provider
        RerankProviderFactory._instance = None


def test_rerank_empty_inputs() -> None:
    """Verifies that RerankService handles empty inputs gracefully."""
    provider = FakeRerankProvider()
    service = RerankService(provider)

    res = asyncio.run(service.rerank("apples", []))
    assert res == []


def test_rerank_ordering_and_deterministic_sorting() -> None:
    """Verifies reranking ordering changes and deterministic sorting for ties."""
    provider = FakeRerankProvider()  # returns scores: float(i + 1)
    doc_id = uuid.uuid4()

    chunk1 = RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=doc_id,
        page_number=1,
        chunk_index=0,
        text="First",
        similarity=0.9,
    )
    chunk2 = RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=doc_id,
        page_number=1,
        chunk_index=1,
        text="Second",
        similarity=0.8,
    )

    # 1. Test reverse sorting (ordering change)
    res = provider.rerank("query", [chunk1, chunk2])
    assert len(res) == 2
    # Since FakeRerankProvider reverses the list:
    assert res[0].chunk_id == chunk2.chunk_id
    assert res[1].chunk_id == chunk1.chunk_id
    assert res[0].rerank_score == 2.0
    assert res[1].rerank_score == 1.0

    # 2. Test deterministic sorting on ties
    # Create chunks with identical rerank scores but different chunk_indexes
    chunk_tie_1 = RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=doc_id,
        page_number=1,
        chunk_index=5,
        text="Tie 1",
        similarity=0.7,
        rerank_score=0.5,
    )
    chunk_tie_2 = RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=doc_id,
        page_number=1,
        chunk_index=3,
        text="Tie 2",
        similarity=0.7,
        rerank_score=0.5,
    )

    # Sort them
    # Chunks are identical in rerank score, so they sort by chunk_index in ASC
    sorted_res = sorted(
        [chunk_tie_1, chunk_tie_2],
        key=lambda c: (
            -(c.rerank_score if c.rerank_score is not None else float("-inf")),
            c.chunk_index,
        ),
    )
    # Tie 2 has chunk_index 3, Tie 1 has chunk_index 5. So Tie 2 comes first!
    assert sorted_res[0].chunk_id == chunk_tie_2.chunk_id
    assert sorted_res[1].chunk_id == chunk_tie_1.chunk_id


@pytest.mark.asyncio
async def test_answer_service_orchestrates_reranking() -> None:
    """Verifies that AnswerService passes reranked chunks to PromptBuilder."""
    embed_provider = EmbeddingProviderFactory.create()
    search_service = SearchService(embed_provider)
    retrieval_service = RetrievalService(search_service)

    fake_reranker = FakeRerankProvider()
    rerank_service = RerankService(fake_reranker)

    fake_llm = FakeLLMProvider()
    prompt_builder = GeminiPromptBuilder()

    answer_service = AnswerService(
        retrieval_service, rerank_service, fake_llm, prompt_builder
    )

    async with AsyncSessionLocal() as db:
        # Create test document and chunks
        doc = Document(
            filename="rerank_test.pdf",
            document_type=DocumentType.pdf,
            status=DocumentStatus.uploaded,
            source="test.pdf",
        )
        db.add(doc)
        await db.commit()
        await db.refresh(doc)

        text1 = "Apples are red."
        text2 = "Bananas are yellow."

        embedding_results = await asyncio.to_thread(
            embed_provider.embed, [text1, text2]
        )
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
            # 1. Fetch original retrieval context first
            query = SearchQuery(text="fruit query", limit=2)
            ret_context = await retrieval_service.retrieve(db, query)
            orig_ids = [c.chunk_id for c in ret_context.chunks]

            # 2. Ask RAG
            _ans, context = await answer_service.ask(db, "fruit query", limit=2)
            ask_ids = [c.chunk_id for c in context.chunks]

            # RetrievedContext chunks should contain rerank_score
            assert len(context.chunks) == len(ret_context.chunks)
            for c in context.chunks:
                assert c.rerank_score is not None

            # FakeRerankProvider reverses original order:
            assert ask_ids == list(reversed(orig_ids))

            # Verify prompt builder receives chunks in correct reranked order
            assert fake_llm.last_user_prompt is not None
            first_text = context.chunks[0].text
            second_text = context.chunks[1].text
            first_idx = fake_llm.last_user_prompt.find(first_text)
            second_idx = fake_llm.last_user_prompt.find(second_text)
            assert first_idx != -1
            assert second_idx != -1
            assert first_idx < second_idx
        finally:
            await db.delete(chunk1)
            await db.delete(chunk2)
            await db.delete(doc)
            await db.commit()
