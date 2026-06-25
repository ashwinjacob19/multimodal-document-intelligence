import os
import uuid

import httpx
import pytest
from sqlalchemy import select

from app.chunking.models import ChunkData
from app.chunking.splitter import TextChunker
from app.db.models.chunk import Chunk
from app.db.models.document import Document, DocumentStatus
from app.db.session import AsyncSessionLocal
from app.main import app
from app.parsers.pdf_parser import Page


def test_text_chunker_unit() -> None:
    """Unit test for TextChunker to verify splitting, index incrementing,

    char count calculation, and page number preservation.
    """
    chunker = TextChunker()
    doc_id = uuid.uuid4()

    # Create dummy pages with some large text to trigger splitting
    pages = [
        Page(
            page=1,
            text=(
                "This is page one text. It contains a lot of sentences to make sure "
                "that the RecursiveCharacterTextSplitter will split it. "
            )
            * 20,
        ),  # ~2100 characters
        Page(page=2, text="This is page two text. It is short."),
    ]

    chunks = chunker.chunk(pages, doc_id)

    # We expect page 1 to be split into at least 2 chunks,
    # and page 2 to have at least 1 chunk.
    assert len(chunks) >= 3

    # Check properties
    for i, c in enumerate(chunks):
        assert isinstance(c, ChunkData)
        assert c.document_id == doc_id
        assert c.chunk_index == i  # Should increment sequentially
        assert len(c.text) == c.char_count  # Correct char count

    # Page number preservation checks
    # The first chunk should belong to page 1
    assert chunks[0].page_number == 1
    # The last chunk should belong to page 2
    assert chunks[-1].page_number == 2


@pytest.mark.asyncio
async def test_pdf_upload_chunking_integration() -> None:
    """Integration test that uploads a sample PDF, processes it,

    and verifies that pages are parsed, chunks are generated,
    and database records are persisted correctly.
    """
    pdf_path = "/app/test.pdf"
    assert os.path.exists(pdf_path), "test.pdf must exist for integration test"

    with open(pdf_path, "rb") as f:
        file_content = f.read()

    # Trigger upload and processing flow
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.post(
            "/documents/upload",
            files={"file": ("test.pdf", file_content, "application/pdf")},
        )

    assert response.status_code == 201
    res_data = response.json()
    assert "document_id" in res_data
    assert res_data["filename"] == "test.pdf"
    assert res_data["status"] == "processing"

    doc_id = uuid.UUID(res_data["document_id"])

    # Verify database state
    async with AsyncSessionLocal() as db:
        # 1. Verify document exists and status remains 'processing'
        stmt_doc = select(Document).where(Document.id == doc_id)
        res_doc = await db.execute(stmt_doc)
        db_doc = res_doc.scalar_one_or_none()

        assert db_doc is not None
        assert db_doc.status == DocumentStatus.processing

        # 2. Verify chunks are persisted
        stmt_chunks = (
            select(Chunk)
            .where(Chunk.document_id == doc_id)
            .order_by(Chunk.chunk_index)
        )
        res_chunks = await db.execute(stmt_chunks)
        db_chunks = res_chunks.scalars().all()

        assert len(db_chunks) > 0

        # Verify sequential indices and correct details
        for i, chunk in enumerate(db_chunks):
            assert chunk.chunk_index == i
            assert chunk.char_count == len(chunk.text)
            assert chunk.page_number >= 1

        # Clean up created database records to maintain test isolation
        await db.delete(db_doc)
        await db.commit()
