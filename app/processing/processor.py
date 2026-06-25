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
            from app.parsers.pdf_parser import PDFParser

            parser = PDFParser()
            pages = parser.extract(document.source)
            page_count = len(pages)
            char_count = sum(len(p.text) for p in pages)
            msg = (
                f"PDF extraction complete for {document.filename}. "
                f"Extracted {page_count} pages and {char_count} characters."
            )
        elif document.document_type == DocumentType.docx:
            msg = "DOCX processing not implemented yet"
        elif document.document_type == DocumentType.image:
            msg = "IMAGE processing not implemented yet"
        elif document.document_type == DocumentType.url:
            msg = "URL processing not implemented yet"
        else:
            msg = (
                f"Unknown document type {document.document_type} "
                "processing not implemented yet"
            )

        logger.info(msg)
        print(msg, flush=True)
