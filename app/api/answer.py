import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.retrieval import (
    RetrievalMetadataResponse,
    RetrievedChunkResponse,
    get_retrieval_service,
)
from app.config import settings
from app.db.session import get_db
from app.llm.factory import LLMProviderFactory
from app.llm.prompt_builder import GeminiPromptBuilder, PromptBuilder
from app.llm.provider import LLMProvider
from app.reranking.factory import RerankProviderFactory
from app.reranking.provider import RerankProvider
from app.services.answer_service import AnswerService
from app.services.rerank_service import RerankService
from app.services.retrieval_service import RetrievalService

router = APIRouter(tags=["ask"])


class AskRequest(BaseModel):
    question: str = Field(..., description="User question for RAG answering")
    document_id: uuid.UUID | None = Field(None, description="Scope filter")
    limit: int | None = Field(None, description="Retrieval limit", ge=1)


class AskResponse(BaseModel):
    answer: str
    metadata: RetrievalMetadataResponse
    chunks: list[RetrievedChunkResponse]


def get_llm_provider() -> LLMProvider:
    """Dependency injection resolver for LLMProvider."""
    return LLMProviderFactory.create()


def get_prompt_builder() -> PromptBuilder:
    """Dependency injection resolver for PromptBuilder."""
    return GeminiPromptBuilder()


def get_rerank_provider() -> RerankProvider:
    """Dependency injection resolver for RerankProvider."""
    return RerankProviderFactory.create()


def get_rerank_service(
    provider: RerankProvider = Depends(get_rerank_provider),
) -> RerankService:
    """Dependency injection resolver for RerankService."""
    return RerankService(provider)


def get_answer_service(
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    rerank_service: RerankService = Depends(get_rerank_service),
    llm_provider: LLMProvider = Depends(get_llm_provider),
    prompt_builder: PromptBuilder = Depends(get_prompt_builder),
) -> AnswerService:
    """Dependency injection resolver for AnswerService."""
    return AnswerService(
        retrieval_service, rerank_service, llm_provider, prompt_builder
    )


@router.post("/ask", response_model=AskResponse)
async def ask(
    request: AskRequest,
    db: AsyncSession = Depends(get_db),
    answer_service: AnswerService = Depends(get_answer_service),
) -> AskResponse:
    """FastAPI endpoint to execute grounded Q&A over document chunks."""
    requested_limit = (
        request.limit
        if request.limit is not None
        else settings.DEFAULT_SEARCH_LIMIT
    )
    clamped_limit = max(1, min(requested_limit, settings.MAX_SEARCH_LIMIT))

    answer, context = await answer_service.ask(
        db=db,
        question=request.question,
        document_id=request.document_id,
        limit=clamped_limit,
    )

    return AskResponse(
        answer=answer,
        metadata=RetrievalMetadataResponse.model_validate(context.metadata),
        chunks=[
            RetrievedChunkResponse.model_validate(chunk)
            for chunk in context.chunks
        ],
    )
