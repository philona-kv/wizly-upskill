"""Owner memories, conversation summaries, workflow executions.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIM = 1536


def upgrade() -> None:
    op.create_table(
        "owner_memories",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("owner_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("memory_type", sa.String(64), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("normalized_content", sa.Text(), nullable=False),
        sa.Column("source_type", sa.String(64), nullable=False),
        sa.Column("source_conversation_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "source_message_ids",
            postgresql.ARRAY(postgresql.UUID(as_uuid=True)),
            nullable=False,
            server_default=sa.text("'{}'::uuid[]"),
        ),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("importance", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("status", sa.String(32), nullable=False, server_default="active"),
        sa.Column("supersedes_memory_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column("embedding_model", sa.String(128), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("last_accessed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.ForeignKeyConstraint(["source_conversation_id"], ["conversations.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["supersedes_memory_id"], ["owner_memories.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "memory_type IN ("
            "'user_preference', 'user_provided_fact', 'project_context', "
            "'ongoing_goal', 'decision', 'unresolved_question', 'conversation_summary', 'other'"
            ")",
            name="owner_memories_type_valid",
        ),
        sa.CheckConstraint(
            "source_type IN ('user_statement', 'assistant_inference', 'system_summary')",
            name="owner_memories_source_type_valid",
        ),
        sa.CheckConstraint(
            "status IN ('active', 'superseded', 'deleted')",
            name="owner_memories_status_valid",
        ),
    )
    op.create_index("ix_owner_memories_owner_id_status", "owner_memories", ["owner_id", "status"])
    op.create_index(
        "ix_owner_memories_embedding_hnsw",
        "owner_memories",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    op.create_table(
        "conversation_summaries",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("owner_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("summary_text", sa.Text(), nullable=False),
        sa.Column("source_message_start_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("source_message_end_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("summarized_message_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("conversation_id", name="uq_conversation_summaries_conversation_id"),
    )
    op.create_index(
        "ix_conversation_summaries_owner_id",
        "conversation_summaries",
        ["owner_id"],
    )

    op.create_table(
        "workflow_executions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("owner_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_message_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("client_message_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("thread_id", sa.String(128), nullable=False),
        sa.Column("graph_version", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("current_node", sa.String(64), nullable=True),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_error_code", sa.String(64), nullable=True),
        sa.Column(
            "started_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'waiting_for_retry', 'completed', 'failed', 'cancelled')",
            name="workflow_executions_status_valid",
        ),
    )
    op.create_index(
        "ix_workflow_executions_owner_conversation_updated",
        "workflow_executions",
        ["owner_id", "conversation_id", "updated_at"],
    )
    op.create_index(
        "uq_workflow_executions_completed_client_message",
        "workflow_executions",
        ["conversation_id", "client_message_id"],
        unique=True,
        postgresql_where=sa.text(
            "client_message_id IS NOT NULL AND status = 'completed'"
        ),
    )


def downgrade() -> None:
    op.drop_index("uq_workflow_executions_completed_client_message", table_name="workflow_executions")
    op.drop_index("ix_workflow_executions_owner_conversation_updated", table_name="workflow_executions")
    op.drop_table("workflow_executions")
    op.drop_index("ix_conversation_summaries_owner_id", table_name="conversation_summaries")
    op.drop_table("conversation_summaries")
    op.drop_index("ix_owner_memories_embedding_hnsw", table_name="owner_memories")
    op.drop_index("ix_owner_memories_owner_id_status", table_name="owner_memories")
    op.drop_table("owner_memories")
