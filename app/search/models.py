from dataclasses import dataclass
import uuid

from app.config import settings


@dataclass(frozen=True)
class SearchQuery:
    text: str
    document_id: uuid.UUID | None = None
    limit: int = settings.DEFAULT_SEARCH_LIMIT


@dataclass(frozen=True)
class SearchResult:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    page_number: int
    chunk_index: int
    text: str
    similarity: float
