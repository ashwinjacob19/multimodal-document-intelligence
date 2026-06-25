import pytest
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.chunking.models import ChunkData
from app.db.models.chunk import Chunk
from app.db.models.document import Document, DocumentStatus, DocumentType
from app.db.session import AsyncSessionLocal
from app.services.chunk_service import ChunkService


@pytest.mark.asyncio
async def test_chunk_persistence() -> None:
    """Verifies that Document and Chunks can be created, saved using ChunkService,

    and that the relationship works correctly.
    """
    async with AsyncSessionLocal() as db:
        # 1. Create a test Document
        doc = Document(
            filename="test_milestone6.pdf",
            document_type=DocumentType.pdf,
            status=DocumentStatus.uploaded,
            source="/tmp/test_milestone6.pdf",
        )
        db.add(doc)
        await db.commit()
        await db.refresh(doc)

        doc_id = doc.id

        try:
            # 2. Create two ChunkData objects
            chunk1 = ChunkData(
                document_id=doc_id,
                page_number=1,
                chunk_index=0,
                text="This is page 1 first chunk.",
                char_count=len("This is page 1 first chunk."),
            )
            chunk2 = ChunkData(
                document_id=doc_id,
                page_number=2,
                chunk_index=1,
                text="This is page 2 first chunk.",
                char_count=len("This is page 2 first chunk."),
            )

            # 3. Save them using ChunkService.save
            db_chunks = await ChunkService.save(db, [chunk1, chunk2])

            # 4. Verify they were successfully persisted
            stmt = (
                select(Chunk)
                .where(Chunk.document_id == doc_id)
                .order_by(Chunk.page_number)
            )
            res = await db.execute(stmt)
            persisted_chunks = res.scalars().all()

            assert len(persisted_chunks) == 2
            assert persisted_chunks[0].chunk_index == 0
            assert persisted_chunks[0].page_number == 1
            assert persisted_chunks[0].text == "This is page 1 first chunk."
            assert persisted_chunks[1].chunk_index == 1
            assert persisted_chunks[1].page_number == 2
            assert persisted_chunks[1].text == "This is page 2 first chunk."

            # 5. Verify relationship with the parent Document
            stmt_doc = (
                select(Document)
                .options(selectinload(Document.chunks))
                .where(Document.id == doc_id)
            )
            res_doc = await db.execute(stmt_doc)
            db_doc = res_doc.scalar_one()

            assert len(db_doc.chunks) == 2
            assert db_doc.chunks[0].id == db_chunks[0].id
            assert db_doc.chunks[1].id == db_chunks[1].id

        finally:
            # Cleanup to keep database clean (cascades to chunks)
            await db.delete(doc)
            await db.commit()
