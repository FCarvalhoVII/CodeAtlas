"""Assemble grounded answering from application configuration."""

from codeatlas.answering.context import ContextAssembler
from codeatlas.answering.service import AnswerService
from codeatlas.core.config import CodeAtlasSettings
from codeatlas.providers.factory import create_llm_provider
from codeatlas.retrieval.factory import create_search_service


def create_answer_service(settings: CodeAtlasSettings) -> AnswerService:
    """Create retrieval, context assembly, and LLM generation as one service."""

    return AnswerService(
        create_search_service(settings),
        create_llm_provider(settings),
        context_assembler=ContextAssembler(
            max_chars=settings.answer.max_context_chars,
            max_tokens=settings.answer.max_context_tokens,
            max_evidence_chars=settings.answer.max_evidence_chars,
        ),
        timeout=settings.answer.timeout_seconds,
    )
