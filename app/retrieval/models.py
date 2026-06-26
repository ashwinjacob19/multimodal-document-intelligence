import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    page_number: int
    chunk_index: int
    text: str
    similarity: float


@dataclass(frozen=True)
class RetrievalMetadata:
    query: str
    retrieved_count: int
    document_id: uuid.UUID | None
    limit: int


@dataclass(frozen=True)
class RetrievalContext:
    chunks: list[RetrievedChunk]
    metadata: RetrievalMetadata
