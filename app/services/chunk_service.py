from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.chunk import Chunk


class ChunkService:
    @staticmethod
    async def save(db: AsyncSession, chunks: list[Chunk]) -> list[Chunk]:
        """Persists a list of Chunk model objects to the PostgreSQL database."""
        db.add_all(chunks)
        await db.commit()
        for chunk in chunks:
            await db.refresh(chunk)
        return chunks
