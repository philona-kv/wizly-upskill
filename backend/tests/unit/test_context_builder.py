import uuid

from app.services.conversation_context_builder import ConversationContextBuilder
from app.services.memory_service import RetrievedMemory
from app.workflows.schemas import CompactMessage


def test_context_builder_labels_and_budget() -> None:
    builder = ConversationContextBuilder(token_budget=500, embedding_model="text-embedding-3-small")
    mem = RetrievedMemory(
        id=uuid.uuid4(),
        memory_type="user_preference",
        content="Prefers concise answers",
        source_type="user_statement",
        source_conversation_id=None,
    )
    assembled = builder.build(
        question="Follow up",
        recent_messages=[
            CompactMessage(id=uuid.uuid4(), role="user", content="Hello"),
        ],
        summary_text="Discussed onboarding docs.",
        memories=[mem],
        exclude_message_id=None,
    )
    assert "not document evidence" in assembled.combined_untrusted
    assert "Persistent memories" in assembled.combined_untrusted
    assert assembled.token_estimate <= 500
