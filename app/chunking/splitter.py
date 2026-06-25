import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.chunking.models import ChunkData
from app.config import settings
from app.parsers.pdf_parser import Page


class TextChunker:
    def __init__(self) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ],
        )

    def chunk(self, pages: list[Page], document_id: uuid.UUID) -> list[ChunkData]:
        """Chunks a list of Page objects and returns a list of ChunkData."""
        chunks = []
        chunk_index = 0
        for page in pages:
            split_texts = self.splitter.split_text(page.text)
            for text in split_texts:
                chunks.append(
                    ChunkData(
                        document_id=document_id,
                        page_number=page.page,
                        chunk_index=chunk_index,
                        text=text,
                        char_count=len(text),
                    )
                )
                chunk_index += 1
        return chunks
