import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.document import Document, DocumentStatus, DocumentType

logger = logging.getLogger(__name__)


class DocumentProcessor:
    @staticmethod
    async def process(document: Document, db: AsyncSession) -> None:
        """Processes a document and updates its status to processing."""
        logger.info("Initiating processing for Document ID: %s", document.id)

        # 1. Update the document status from uploaded to processing
        document.status = DocumentStatus.processing
        await db.commit()
        await db.refresh(document)

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

            # Persist the chunks
            await ChunkService.save(db, chunks)

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

        elif document.document_type == DocumentType.docx:
            msg = "DOCX processing not implemented yet"
            logger.info(msg)
            print(msg, flush=True)
        elif document.document_type == DocumentType.image:
            msg = "IMAGE processing not implemented yet"
            logger.info(msg)
            print(msg, flush=True)
        elif document.document_type == DocumentType.url:
            msg = "URL processing not implemented yet"
            logger.info(msg)
            print(msg, flush=True)
        else:
            msg = (
                f"Unknown document type {document.document_type} "
                "processing not implemented yet"
            )
            logger.info(msg)
            print(msg, flush=True)
