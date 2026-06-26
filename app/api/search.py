import uuid

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.session import get_db
from app.embeddings.factory import EmbeddingProviderFactory
from app.embeddings.provider import EmbeddingProvider
from app.search.models import SearchQuery
from app.services.search_service import SearchService

router = APIRouter(prefix="/search", tags=["search"])


class SearchResultResponse(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    page_number: int
    chunk_index: int
    text: str
    similarity: float

    model_config = {
        "from_attributes": True,
    }


def get_embedding_provider() -> EmbeddingProvider:
    return EmbeddingProviderFactory.create()


def get_search_service(
    provider: EmbeddingProvider = Depends(get_embedding_provider),
) -> SearchService:
    return SearchService(provider)


@router.get("/chunks", response_model=list[SearchResultResponse])
async def search_chunks(
    q: str = Query(..., description="The query text for semantic search"),
    document_id: uuid.UUID | None = Query(
        None, description="Optional document ID filter"
    ),
    limit: int = Query(
        settings.DEFAULT_SEARCH_LIMIT,
        description=(
            "Maximum number of results to return "
            f"(max {settings.MAX_SEARCH_LIMIT})"
        ),
        ge=1,
    ),
    db: AsyncSession = Depends(get_db),
    search_service: SearchService = Depends(get_search_service),
) -> list[SearchResultResponse]:
    """Semantic vector search endpoint to retrieve the most relevant text chunks."""
    # Clamp requested limit parameter to settings.MAX_SEARCH_LIMIT
    clamped_limit = min(limit, settings.MAX_SEARCH_LIMIT)

    query = SearchQuery(
        text=q,
        document_id=document_id,
        limit=clamped_limit,
    )

    results = await search_service.search(db, query)
    return [SearchResultResponse.model_validate(res) for res in results]
