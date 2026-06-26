from abc import ABC, abstractmethod

from app.llm.models import PromptBundle
from app.retrieval.models import RetrievalContext


class PromptBuilder(ABC):
    @abstractmethod
    def build(self, question: str, context: RetrievalContext) -> PromptBundle:
        """Constructs a system/user prompt bundle based on context and question."""
        pass


class GeminiPromptBuilder(PromptBuilder):
    def build(self, question: str, context: RetrievalContext) -> PromptBundle:
        """Builds a system/user prompt bundle tailored for Gemini."""
        system_prompt = (
            "You are an assistant that answers questions based strictly on the "
            "provided context.\n"
            "If the answer cannot be determined from the context, state clearly "
            "that you do not know.\n"
            "Cite your sources using chunk index/number where appropriate."
        )

        user_parts = ["Retrieved Context Chunks:"]
        for idx, chunk in enumerate(context.chunks, 1):
            user_parts.append(
                f"[{idx}] (Page {chunk.page_number}, Doc: {chunk.document_id}): "
                f"{chunk.text}"
            )
        user_parts.append(f"\nUser Question: {question}")

        return PromptBundle(
            system_prompt=system_prompt,
            user_prompt="\n".join(user_parts),
        )
