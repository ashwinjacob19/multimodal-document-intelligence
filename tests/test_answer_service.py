import asyncio
import uuid

import httpx
import pytest

from app.db.models.chunk import Chunk
from app.db.models.document import Document, DocumentStatus, DocumentType
from app.db.session import AsyncSessionLocal
from app.embeddings.factory import EmbeddingProviderFactory
from app.llm.models import PromptBundle
from app.llm.prompt_builder import GeminiPromptBuilder
from app.llm.provider import FakeLLMProvider
from app.main import app
from app.retrieval.models import RetrievalContext, RetrievalMetadata, RetrievedChunk
from app.services.answer_service import AnswerService
from app.services.retrieval_service import RetrievalService
from app.services.search_service import SearchService


def test_gemini_prompt_builder() -> None:
    """Verifies GeminiPromptBuilder returns a valid PromptBundle."""
    builder = GeminiPromptBuilder()
    doc_id = uuid.uuid4()
    chunk1 = RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=doc_id,
        page_number=1,
        chunk_index=0,
        text="Apples are red.",
        similarity=0.9,
    )
    metadata = RetrievalMetadata(
        query="apples",
        retrieved_count=1,
        document_id=None,
        limit=5,
    )
    context = RetrievalContext(chunks=[chunk1], metadata=metadata)
    bundle = builder.build("apples query", context)

    assert isinstance(bundle, PromptBundle)
    assert "strictly on the provided context" in bundle.system_prompt
    assert "Apples are red." in bundle.user_prompt
    assert "apples query" in bundle.user_prompt


@pytest.mark.asyncio
async def test_answer_service_orchestration() -> None:
    """Verifies AnswerService orchestration and prompt generation."""
    embed_provider = EmbeddingProviderFactory.create()
    search_service = SearchService(embed_provider)
    retrieval_service = RetrievalService(search_service)
    fake_llm = FakeLLMProvider(default_response="Bananas are sweet.")
    prompt_builder = GeminiPromptBuilder()
    answer_service = AnswerService(retrieval_service, fake_llm, prompt_builder)

    async with AsyncSessionLocal() as db:
        # Create test document and chunk
        doc = Document(
            filename="answer_test.pdf",
            document_type=DocumentType.pdf,
            status=DocumentStatus.uploaded,
            source="test.pdf",
        )
        db.add(doc)
        await db.commit()
        await db.refresh(doc)

        text = "Bananas are yellow."
        embedding_results = await asyncio.to_thread(
            embed_provider.embed, [text]
        )
        emb = embedding_results[0].embedding

        chunk = Chunk(
            document_id=doc.id,
            page_number=1,
            chunk_index=0,
            text=text,
            char_count=len(text),
            embedding=emb,
        )
        db.add(chunk)
        await db.commit()

        try:
            ans, context = await answer_service.ask(db, "yellow fruit", limit=1)
            assert ans == "Bananas are sweet."
            assert len(context.chunks) == 1
            assert context.chunks[0].text == "Bananas are yellow."

            # Verify prompts were recorded correctly in the fake LLM
            assert fake_llm.last_system_prompt is not None
            assert "Bananas are yellow." in fake_llm.last_user_prompt
            assert "yellow fruit" in fake_llm.last_user_prompt
        finally:
            await db.delete(chunk)
            await db.delete(doc)
            await db.commit()


@pytest.mark.asyncio
async def test_answer_service_empty_retrieval() -> None:
    """Verifies RAG works correctly when no chunks are retrieved."""
    embed_provider = EmbeddingProviderFactory.create()
    search_service = SearchService(embed_provider)
    retrieval_service = RetrievalService(search_service)
    fake_llm = FakeLLMProvider(default_response="I do not know.")
    prompt_builder = GeminiPromptBuilder()
    answer_service = AnswerService(retrieval_service, fake_llm, prompt_builder)

    async with AsyncSessionLocal() as db:
        ans, context = await answer_service.ask(db, "", limit=1)
        assert ans == "I do not know."
        assert len(context.chunks) == 0


@pytest.mark.asyncio
async def test_answer_api_clamping_and_mocking() -> None:
    """Verifies endpoint serialization and limit parameter clamping."""
    from app.api.answer import get_llm_provider

    fake_llm = FakeLLMProvider(default_response="Mocked API answer.")
    app.dependency_overrides[get_llm_provider] = lambda: fake_llm

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as ac:
            # 1. Ask endpoint call
            response = await ac.post(
                "/ask",
                json={"question": "apples", "limit": 100},
            )
            assert response.status_code == 200
            res_data = response.json()
            assert res_data["answer"] == "Mocked API answer."
            assert "chunks" in res_data
            # Clamped to MAX_SEARCH_LIMIT (20)
            assert res_data["metadata"]["limit"] == 20
            assert res_data["metadata"]["query"] == "apples"
    finally:
        app.dependency_overrides.clear()


def test_llm_provider_factory_selection() -> None:
    """Verifies that LLMProviderFactory yields correct provider."""
    from app.config import settings
    from app.llm.factory import LLMProviderFactory
    from app.llm.provider import FakeLLMProvider, GeminiProvider

    # Reset singleton state
    LLMProviderFactory._instance = None

    # Scenario 1: Key not configured
    original_key = settings.GEMINI_API_KEY
    settings.GEMINI_API_KEY = None
    try:
        provider = LLMProviderFactory.create()
        assert isinstance(provider, FakeLLMProvider)
    finally:
        LLMProviderFactory._instance = None

    # Scenario 2: Key configured
    settings.GEMINI_API_KEY = "test_gemini_key"
    try:
        provider = LLMProviderFactory.create()
        assert isinstance(provider, GeminiProvider)
        assert provider.api_key == "test_gemini_key"
    finally:
        # Restore settings
        settings.GEMINI_API_KEY = original_key
        LLMProviderFactory._instance = None
