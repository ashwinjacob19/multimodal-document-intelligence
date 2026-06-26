import uuid

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.search import get_search_service
from app.config import settings
from app.db.session import get_db
from app.search.models import SearchQuery
from app.services.retrieval_service import RetrievalService
from app.services.search_service import SearchService

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


class RetrievedChunkResponse(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    page_number: int
    chunk_index: int
    text: str
    similarity: float
    rerank_score: float | None = None

    model_config = {
        "from_attributes": True,
    }


class RetrievalMetadataResponse(BaseModel):
    query: str
    retrieved_count: int
    document_id: uuid.UUID | None
    limit: int

    model_config = {
        "from_attributes": True,
    }


class RetrievalContextResponse(BaseModel):
    chunks: list[RetrievedChunkResponse]
    metadata: RetrievalMetadataResponse

    model_config = {
        "from_attributes": True,
    }


def get_retrieval_service(
    search_service: SearchService = Depends(get_search_service),
) -> RetrievalService:
    """Dependency resolver returning a RetrievalService instance."""
    return RetrievalService(search_service)


@router.get("", response_model=RetrievalContextResponse)
async def retrieve_context(
    q: str = Query(..., description="The query text for semantic retrieval"),
    document_id: uuid.UUID | None = Query(
        None, description="Optional document ID scope filter"
    ),
    limit: int = Query(
        settings.DEFAULT_SEARCH_LIMIT,
        description=(
            "Maximum number of chunks to retrieve "
            f"(max {settings.MAX_SEARCH_LIMIT})"
        ),
        ge=1,
    ),
    db: AsyncSession = Depends(get_db),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
) -> RetrievalContextResponse:
    """Semantic context retrieval endpoint to package relevant document chunks."""
    # Clamp requested limit parameter to settings.MAX_SEARCH_LIMIT and at least 1
    clamped_limit = max(1, min(limit, settings.MAX_SEARCH_LIMIT))

    query = SearchQuery(
        text=q,
        document_id=document_id,
        limit=clamped_limit,
    )

    context = await retrieval_service.retrieve(db, query)
    return RetrievalContextResponse.model_validate(context)
