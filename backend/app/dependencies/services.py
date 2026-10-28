"""Service wiring. Long-lived clients live on ``app.state`` (created in the lifespan);
these dependencies assemble per-request service objects around them. Tests override
them via ``app.dependency_overrides``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import BackgroundTasks, Depends, Request
from openai import AsyncOpenAI

from app.core.config import ConfigurationError, Settings
from app.dependencies.auth import SettingsDep
from app.dependencies.database import SessionFactoryDep
from app.services.answer_repair import AnswerRepairService
from app.services.chat_postprocess import ChatPostProcessService
from app.services.chat_service import ChatService
from app.services.chunking_service import ChunkingService
from app.services.pdf_extractor import PdfExtractOptions
from app.services.context_builder import ContextBuilder
from app.services.conversation_context_builder import ConversationContextBuilder
from app.services.conversation_summary_service import ConversationSummaryService
from app.services.document_service import DocumentService
from app.services.embedding_service import Embedder
from app.services.evidence_assessor import HeuristicEvidenceAssessor, OpenAIEvidenceAssessor
from app.services.generation_service import Generator
from app.services.ingestion_service import IngestionService
from app.services.job_runner import BackgroundTasksJobRunner, IngestionJobRunner
from app.services.memory_extraction_service import MemoryExtractionService
from app.services.memory_service import MemoryService
from app.services.query_planner import HeuristicQueryPlanner, OpenAIQueryPlanner
from app.services.retrieval_service import RetrievalService
from app.services.storage_service import StorageService
from app.services.twin_workflow_runner import TwinWorkflowRunner
from app.services.workflow_execution_service import WorkflowExecutionService
from pathlib import Path

from app.services.document_promotion_service import DocumentPromotionService
from app.services.playground_evaluation_service import PlaygroundEvaluationService
from app.services.playground_experiment_service import PlaygroundExperimentService
from app.services.playground_query_service import PlaygroundQueryService
from app.services.playground_workflow_runner import PlaygroundWorkflowRunner
from app.services.prompt_template_service import PromptTemplateService
from app.workflows.graph import build_playground_graph, build_twin_graph
from app.workflows.nodes import WorkflowDeps
from app.workflows.tools import ConversationHistoryTool, DocumentSearchTool


def get_storage(request: Request) -> StorageService | None:
    return request.app.state.storage


def get_embedder(request: Request) -> Embedder:
    embedder = request.app.state.embedder
    if embedder is None:
        raise ConfigurationError("OPENAI_API_KEY is not configured")
    return embedder


def get_document_service(
    settings: SettingsDep,
    session_factory: SessionFactoryDep,
    storage: Annotated[StorageService | None, Depends(get_storage)],
) -> DocumentService:
    return DocumentService(
        session_factory, storage, max_upload_bytes=settings.max_upload_size_bytes
    )


def get_ingestion_service(
    settings: SettingsDep,
    session_factory: SessionFactoryDep,
    storage: Annotated[StorageService | None, Depends(get_storage)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
) -> IngestionService:
    if storage is None:
        raise ConfigurationError("Supabase storage is not configured")
    return IngestionService(
        session_factory,
        storage,
        ChunkingService(
            settings.chunk_size_tokens,
            settings.chunk_overlap_tokens,
            embedder,
            breakpoint_percentile=settings.semantic_chunk_breakpoint_percentile,
            encoding_model=settings.openai_embedding_model,
        ),
        embedder,
        stale_after_seconds=settings.ingestion_stale_after_seconds,
        pdf_extract_options=PdfExtractOptions(
            page_ocr_enabled=settings.pdf_page_ocr_enabled,
            page_ocr_scale=settings.pdf_page_ocr_scale,
            page_ocr_min_text_chars=settings.pdf_page_ocr_min_text_chars,
            embedded_image_ocr_enabled=settings.pdf_embedded_image_ocr_enabled,
            embedded_image_min_pixels=settings.pdf_embedded_image_min_pixels,
            max_embedded_images_per_page=settings.pdf_max_embedded_images_per_page,
            table_extraction_enabled=settings.pdf_table_extraction_enabled,
        ),
    )


IngestionServiceDep = Annotated[IngestionService, Depends(get_ingestion_service)]


def get_job_runner(
    background_tasks: BackgroundTasks, ingestion: IngestionServiceDep
) -> IngestionJobRunner:
    return BackgroundTasksJobRunner(background_tasks, ingestion)


def get_generator(request: Request) -> Generator:
    generator = getattr(request.app.state, "generator", None)
    if generator is None:
        raise ConfigurationError("Chat model is not configured")
    return generator


def _llm_client(settings: Settings) -> AsyncOpenAI | None:
    try:
        kwargs: dict = {
            "api_key": settings.require_openai(),
            "timeout": settings.openai_timeout_seconds,
            "max_retries": 0,
        }
        base = settings.embedding_api_base()
        if base is not None:
            kwargs["base_url"] = base
        return AsyncOpenAI(**kwargs)
    except ConfigurationError:
        return None


def get_memory_service(
    settings: SettingsDep,
    session_factory: SessionFactoryDep,
    embedder: Annotated[Embedder, Depends(get_embedder)],
) -> MemoryService:
    return MemoryService(
        session_factory,
        embedder,
        embedding_model=settings.memory_embedding_model_name(),
        top_k=settings.memory_retrieval_top_k,
        min_importance=settings.memory_min_importance,
        max_records=settings.memory_max_records_per_user,
        enabled=settings.memory_enabled,
    )


MemoryServiceDep = Annotated[MemoryService, Depends(get_memory_service)]


def get_summary_service(
    settings: SettingsDep,
    session_factory: SessionFactoryDep,
) -> ConversationSummaryService:
    llm = _llm_client(settings)
    return ConversationSummaryService(
        session_factory,
        llm,
        settings.openai_chat_model,
        enabled=settings.conversation_summary_enabled,
        max_tokens_hint=settings.conversation_summary_max_tokens,
        incremental=settings.conversation_summary_incremental,
    )


SummaryServiceDep = Annotated[ConversationSummaryService, Depends(get_summary_service)]


def get_workflow_execution_service(
    session_factory: SessionFactoryDep,
) -> WorkflowExecutionService:
    return WorkflowExecutionService(session_factory)


WorkflowExecutionServiceDep = Annotated[
    WorkflowExecutionService, Depends(get_workflow_execution_service)
]


def get_chat_service(
    request: Request,
    settings: SettingsDep,
    session_factory: SessionFactoryDep,
    embedder: Annotated[Embedder, Depends(get_embedder)],
    generator: Annotated[Generator, Depends(get_generator)],
    memory_service: MemoryServiceDep,
    summary_service: SummaryServiceDep,
    execution_service: WorkflowExecutionServiceDep,
    background_tasks: BackgroundTasks,
) -> ChatService:
    retrieval = RetrievalService(
        embedder,
        top_k=settings.retrieval_top_k,
        max_candidates=settings.retrieval_max_candidates,
        embedding_model=settings.openai_embedding_model,
    )
    context_builder = ContextBuilder(
        max_tokens=settings.max_context_tokens,
        embedding_model=settings.openai_embedding_model,
    )
    conversation_context_builder = ConversationContextBuilder(
        token_budget=settings.memory_context_token_budget,
        embedding_model=settings.openai_embedding_model,
    )
    llm = _llm_client(settings)
    if llm is not None:
        planner = OpenAIQueryPlanner(llm, settings.openai_chat_model)
        assessor = OpenAIEvidenceAssessor(llm, settings.openai_chat_model)
    else:
        planner = HeuristicQueryPlanner()
        assessor = HeuristicEvidenceAssessor()
    deps = WorkflowDeps(
        planner=planner,
        assessor=assessor,
        generator=generator,
        document_tool=DocumentSearchTool(session_factory, retrieval),
        conversation_tool=ConversationHistoryTool(session_factory),
        context_builder=context_builder,
        answer_repair=AnswerRepairService(generator),
        max_evidence_items=settings.langgraph_max_evidence_items,
        memory_service=memory_service if settings.memory_enabled else None,
        summary_service=summary_service if settings.conversation_summary_enabled else None,
        conversation_context_builder=conversation_context_builder,
    )
    checkpointer = getattr(request.app.state, "langgraph_checkpointer", None)
    if settings.langgraph_checkpointing_enabled and checkpointer is None:
        checkpointer = None
    compiled = build_twin_graph(deps, checkpointer=checkpointer)
    workflow = TwinWorkflowRunner(
        compiled,
        max_retrieval_attempts=settings.langgraph_max_retrieval_attempts,
        max_answer_repairs=settings.langgraph_max_answer_repairs,
        enable_debug=settings.langgraph_enable_debug_metadata,
        graph_version=settings.langgraph_graph_version,
        recovery_enabled=settings.langgraph_recovery_enabled,
        max_recovery_attempts=settings.langgraph_max_recovery_attempts,
        retry_backoff_seconds=settings.langgraph_retry_backoff_seconds,
    )
    extraction = MemoryExtractionService(
        llm,
        settings.openai_chat_model,
        memory_service,
        enabled=settings.memory_extraction_enabled and settings.memory_enabled,
    )
    postprocess = ChatPostProcessService(
        session_factory,
        extraction,
        summary_service,
        memory_interval=settings.memory_extraction_interval_messages,
        summary_trigger=settings.conversation_summary_trigger_messages,
    )
    postprocess.set_scheduler(
        lambda owner_id, conversation_id: background_tasks.add_task(
            postprocess.run, owner_id, conversation_id
        )
    )
    return ChatService(
        session_factory,
        workflow,
        history_limit=settings.max_chat_history_messages,
        enable_debug_metadata=settings.langgraph_enable_debug_metadata,
        execution_service=execution_service,
        graph_version=settings.langgraph_graph_version,
        postprocess=postprocess,
    )


def _workflow_deps(
    settings: Settings,
    session_factory: SessionFactoryDep,
    embedder: Embedder,
    generator: Generator,
) -> WorkflowDeps:
    llm = _llm_client(settings)
    if llm is not None:
        planner = OpenAIQueryPlanner(llm, settings.openai_chat_model)
        assessor = OpenAIEvidenceAssessor(llm, settings.openai_chat_model)
    else:
        planner = HeuristicQueryPlanner()
        assessor = HeuristicEvidenceAssessor()
    retrieval = RetrievalService(
        embedder,
        top_k=settings.retrieval_top_k,
        max_candidates=settings.retrieval_max_candidates,
        embedding_model=settings.openai_embedding_model,
    )
    context_builder = ContextBuilder(
        max_tokens=settings.max_context_tokens,
        embedding_model=settings.openai_embedding_model,
    )
    conversation_context_builder = ConversationContextBuilder(
        token_budget=settings.memory_context_token_budget,
        embedding_model=settings.openai_embedding_model,
    )
    return WorkflowDeps(
        planner=planner,
        assessor=assessor,
        generator=generator,
        document_tool=DocumentSearchTool(session_factory, retrieval),
        conversation_tool=ConversationHistoryTool(session_factory),
        context_builder=context_builder,
        answer_repair=AnswerRepairService(generator),
        max_evidence_items=settings.langgraph_max_evidence_items,
        memory_service=None,
        summary_service=None,
        conversation_context_builder=conversation_context_builder,
    )


def get_playground_query_service(
    request: Request,
    settings: SettingsDep,
    session_factory: SessionFactoryDep,
    embedder: Annotated[Embedder, Depends(get_embedder)],
    generator: Annotated[Generator, Depends(get_generator)],
) -> PlaygroundQueryService:
    deps = _workflow_deps(settings, session_factory, embedder, generator)
    graph = build_playground_graph(deps, checkpointer=None)
    workflow = PlaygroundWorkflowRunner(
        graph,
        max_retrieval_attempts=settings.langgraph_max_retrieval_attempts,
        max_answer_repairs=settings.langgraph_max_answer_repairs,
        enable_debug=settings.langgraph_enable_debug_metadata,
        graph_version="playground-v1",
    )
    return PlaygroundQueryService(
        session_factory,
        workflow,
        PromptTemplateService(session_factory),
    )


PlaygroundQueryServiceDep = Annotated[PlaygroundQueryService, Depends(get_playground_query_service)]


def get_prompt_template_service(
    session_factory: SessionFactoryDep,
) -> PromptTemplateService:
    return PromptTemplateService(session_factory)


PromptTemplateServiceDep = Annotated[PromptTemplateService, Depends(get_prompt_template_service)]


def get_document_promotion_service(
    session_factory: SessionFactoryDep,
) -> DocumentPromotionService:
    return DocumentPromotionService(session_factory)


DocumentPromotionServiceDep = Annotated[
    DocumentPromotionService, Depends(get_document_promotion_service)
]


def get_playground_experiment_service(
    query_service: PlaygroundQueryServiceDep,
    session_factory: SessionFactoryDep,
) -> PlaygroundExperimentService:
    return PlaygroundExperimentService(
        session_factory,
        query_service,
        PromptTemplateService(session_factory),
    )


PlaygroundExperimentServiceDep = Annotated[
    PlaygroundExperimentService, Depends(get_playground_experiment_service)
]


def get_playground_evaluation_service(
    query_service: PlaygroundQueryServiceDep,
    session_factory: SessionFactoryDep,
) -> PlaygroundEvaluationService:
    repo_root = Path(__file__).resolve().parents[3]
    return PlaygroundEvaluationService(session_factory, query_service, repo_root=repo_root)


PlaygroundEvaluationServiceDep = Annotated[
    PlaygroundEvaluationService, Depends(get_playground_evaluation_service)
]


DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
JobRunnerDep = Annotated[IngestionJobRunner, Depends(get_job_runner)]
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
