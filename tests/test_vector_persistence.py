import math

import pytest
from sqlalchemy import select

from app.config import settings
from app.db.models.chunk import Chunk
from app.db.models.document import Document, DocumentStatus, DocumentType
from app.db.session import AsyncSessionLocal
from app.embeddings.factory import EmbeddingProviderFactory
from app.processing.processor import DocumentProcessor


@pytest.mark.asyncio
async def test_vector_persistence_and_lifecycle() -> None:
    """Verifies that PDF processing generates, normalizes, and persists

    chunk embeddings, and updates document status to 'processed'.
    """
    provider = EmbeddingProviderFactory.create()
    processor = DocumentProcessor(provider)

    async with AsyncSessionLocal() as db:
        # 1. Create a test Document using existing test.pdf
        doc = Document(
            filename="test_vector_persistence.pdf",
            document_type=DocumentType.pdf,
            status=DocumentStatus.uploaded,
            source="test.pdf",
        )
        db.add(doc)
        await db.commit()
        await db.refresh(doc)

        try:
            # 2. Run the process pipeline
            await processor.process(doc, db)

            # 3. Refresh and verify status is 'processed'
            await db.refresh(doc)
            assert doc.status == DocumentStatus.processed

            # 4. Fetch the chunks from the database and verify vectors
            stmt = select(Chunk).where(Chunk.document_id == doc.id)
            res = await db.execute(stmt)
            chunks = res.scalars().all()

            assert len(chunks) > 0
            for chunk in chunks:
                assert chunk.embedding is not None
                # Verify dimension matches configured EMBEDDING_DIMENSION
                assert len(chunk.embedding) == settings.EMBEDDING_DIMENSION
                # Verify normalized vectors have approximately unit norm
                l2_norm = math.sqrt(sum(val * val for val in chunk.embedding))
                assert math.isclose(l2_norm, 1.0, abs_tol=1e-3)

        finally:
            # Cleanup document and cascading chunks
            await db.delete(doc)
            await db.commit()


@pytest.mark.asyncio
async def test_processing_failure_lifecycle() -> None:
    """Verifies that processing failures update status to 'failed'

    and re-raise exceptions.
    """
    provider = EmbeddingProviderFactory.create()
    processor = DocumentProcessor(provider)

    async with AsyncSessionLocal() as db:
        # Create a document with a non-existent file path to cause extraction failure
        doc = Document(
            filename="non_existent.pdf",
            document_type=DocumentType.pdf,
            status=DocumentStatus.uploaded,
            source="non_existent.pdf",
        )
        db.add(doc)
        await db.commit()
        await db.refresh(doc)

        try:
            # Verify that process raises FileNotFoundError on missing file
            with pytest.raises(Exception, match="no such file"):
                await processor.process(doc, db)

            # Verify that status was updated once to 'failed'
            await db.refresh(doc)
            assert doc.status == DocumentStatus.failed

        finally:
            # Cleanup
            await db.delete(doc)
            await db.commit()



def test_singleton_provider_stability() -> None:
    """Verifies that the factory continues to return a cached singleton instance."""
    provider1 = EmbeddingProviderFactory.create()
    provider2 = EmbeddingProviderFactory.create()
    assert provider1 is provider2
