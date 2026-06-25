import logging
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.document import Document, DocumentStatus, DocumentType

logger = logging.getLogger(__name__)


class DocumentProcessor:
    @staticmethod
    async def process(document: Document, db: AsyncSession) -> None:
        """Processes a document by changing its status to processing and logging a placeholder message."""
        logger.info("Initiating processing for Document ID: %s", document.id)

        # 1. Update the document status from uploaded to processing
        document.status = DocumentStatus.processing
        await db.commit()
        await db.refresh(document)

        # 2. Determine document type and log/print a message indicating it's not implemented
        if document.document_type == DocumentType.pdf:
            msg = "PDF processing not implemented yet"
        elif document.document_type == DocumentType.docx:
            msg = "DOCX processing not implemented yet"
        elif document.document_type == DocumentType.image:
            msg = "IMAGE processing not implemented yet"
        elif document.document_type == DocumentType.url:
            msg = "URL processing not implemented yet"
        else:
            msg = f"Unknown document type {document.document_type} processing not implemented yet"

        logger.info(msg)
        print(msg, flush=True)
