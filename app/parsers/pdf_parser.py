from dataclasses import dataclass

import fitz  # PyMuPDF


@dataclass
class Page:
    page: int
    text: str


class PDFParser:
    def extract(self, source_path: str) -> list[Page]:
        """Parses a PDF document and extracts plain text page by page.

        Returns:
            A list of Page dataclass objects.
        """
        results = []
        with fitz.open(source_path) as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                results.append(
                    Page(
                        page=page_num + 1,  # 1-based index
                        text=text,
                    )
                )
        return results
