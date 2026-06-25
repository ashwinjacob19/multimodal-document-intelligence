import uuid
from dataclasses import dataclass


@dataclass
class ChunkData:
    document_id: uuid.UUID
    page_number: int
    chunk_index: int
    text: str
    char_count: int
