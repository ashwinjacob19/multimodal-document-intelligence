from sqlalchemy.ext.asyncio import AsyncSession

from app.chunking.models import EmbeddedChunkData
from app.db.models.chunk import Chunk


class ChunkService:
    @staticmethod
    async def save(db: AsyncSession, chunks: list[EmbeddedChunkData]) -> list[Chunk]:
        """Converts EmbeddedChunkData list to Chunk models and persists them."""
        db_chunks = [
            Chunk(
                document_id=c.document_id,
                page_number=c.page_number,
                chunk_index=c.chunk_index,
                text=c.text,
                char_count=c.char_count,
                embedding=c.embedding,
            )
            for c in chunks
        ]
        db.add_all(db_chunks)
        await db.commit()
        for chunk in db_chunks:
            await db.refresh(chunk)
        return db_chunks

