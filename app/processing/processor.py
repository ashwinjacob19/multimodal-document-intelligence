import asyncio
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.chunking.models import EmbeddedChunkData
from app.db.models.document import Document, DocumentStatus, DocumentType
from app.embeddings.provider import EmbeddingProvider

logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self, embedding_provider: EmbeddingProvider):
        """Initializes the processor with an injected EmbeddingProvider dependency."""
        self.embedding_provider = embedding_provider

    async def process(self, document: Document, db: AsyncSession) -> None:
        """Processes a document and updates its status to processing."""
        logger.info("Initiating processing for Document ID: %s", document.id)

        # 1. Update the document status from uploaded to processing
        document.status = DocumentStatus.processing
        await db.commit()
        await db.refresh(document)

        try:
            # 2. Determine document type and process
            if document.document_type == DocumentType.pdf:
                from app.chunking.splitter import TextChunker
                from app.parsers.pdf_parser import PDFParser
                from app.services.chunk_service import ChunkService

                parser = PDFParser()
                pages = parser.extract(document.source)

                # Chunk the pages
                chunker = TextChunker()
                chunks = chunker.chunk(pages, document.id)

                if chunks:
                    # Generate embeddings (CPU-bound local sentence-transformers,
                    # offloaded to thread pool)
                    texts = [c.text for c in chunks]
                    embedding_results = await asyncio.to_thread(
                        self.embedding_provider.embed, texts
                    )

                    # Map ChunkData to EmbeddedChunkData
                    embedded_chunks = [
                        EmbeddedChunkData(
                            document_id=c.document_id,
                            page_number=c.page_number,
                            chunk_index=c.chunk_index,
                            text=c.text,
                            char_count=c.char_count,
                            embedding=emb.embedding,
                        )
                        for c, emb in zip(chunks, embedding_results, strict=True)
                    ]

                    # Persist the embedded chunks
                    await ChunkService.save(db, embedded_chunks)
                else:
                    embedded_chunks = []

                # Log page-by-page progress
                print(f"Processing: {document.filename}\n", flush=True)
                logger.info("Processing: %s", document.filename)

                page_chunk_counts = {}
                for chunk_data in chunks:
                    page_chunk_counts[chunk_data.page_number] = (
                        page_chunk_counts.get(chunk_data.page_number, 0) + 1
                    )

                for page in pages:
                    num_chunks = page_chunk_counts.get(page.page, 0)
                    print(f"Chunking page {page.page}...", flush=True)
                    print(f"Generated {num_chunks} chunks\n", flush=True)
                    logger.info(
                        "Chunking page %d... Generated %d chunks",
                        page.page,
                        num_chunks,
                    )

                # Log summary
                page_count = len(pages)
                chunk_count = len(chunks)
                total_characters = sum(len(p.text) for p in pages)

                summary_lines = [
                    "Document Summary",
                    "----------------",
                    f"Pages: {page_count}",
                    f"Chunks: {chunk_count}",
                    f"Characters: {total_characters}",
                ]
                summary_text = "\n".join(summary_lines)
                print(summary_text, flush=True)
                logger.info("\n" + summary_text)

                # Mark as processed
                document.status = DocumentStatus.processed
                await db.commit()

            elif document.document_type == DocumentType.docx:
                raise NotImplementedError("DOCX processing not implemented yet")
            elif document.document_type == DocumentType.image:
                raise NotImplementedError("IMAGE processing not implemented yet")
            elif document.document_type == DocumentType.url:
                raise NotImplementedError("URL processing not implemented yet")
            else:
                raise NotImplementedError(
                    f"Unknown document type {document.document_type} "
                    "processing not implemented yet"
                )

        except Exception as e:
            logger.error("Processing failed for document %s: %s", document.id, e)
            document.status = DocumentStatus.failed
            await db.commit()
            raise e

