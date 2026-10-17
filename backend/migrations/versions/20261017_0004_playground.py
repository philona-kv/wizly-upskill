"""Playground: prompt templates, experiments, promotion audit.

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-17
"""

from collections.abc import Sequence
import uuid

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SYSTEM_TEMPLATES = [
    (
        uuid.UUID("a0000001-0001-4000-8000-000000000001"),
        "Grounded Q&A",
        "Answer using only supplied evidence; cite pages; abstain when insufficient.",
        "Answer the question using only the evidence below.\n\nQuestion: {question}\n\nEvidence:\n{retrieved_context}\n\n{citation_instructions}",
    ),
    (
        uuid.UUID("a0000002-0002-4000-8000-000000000002"),
        "Structured Extraction",
        "Extract specified fields as JSON; mark missing vs inferred.",
        "Extract structured fields from the evidence. Return valid JSON only.\n\nQuestion: {question}\n\nEvidence:\n{retrieved_context}",
    ),
    (
        uuid.UUID("a0000003-0003-4000-8000-000000000003"),
        "Document Summarization",
        "Summarize key facts and caveats from evidence.",
        "Summarize the evidence for the user question. Note missing information.\n\nQuestion: {question}\n\nEvidence:\n{retrieved_context}",
    ),
    (
        uuid.UUID("a0000004-0004-4000-8000-000000000004"),
        "Comparative Analysis",
        "Compare sources; attribute claims; note contradictions.",
        "Compare evidence across sources. Attribute each claim.\n\nQuestion: {question}\n\nEvidence:\n{retrieved_context}\n\n{citation_instructions}",
    ),
]


def upgrade() -> None:
    op.create_table(
        "prompt_templates",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("owner_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("current_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("is_system", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_prompt_templates_owner_id", "prompt_templates", ["owner_id"])

    op.create_table(
        "prompt_template_versions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("template_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("template_text", sa.Text(), nullable=False),
        sa.Column("config", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["template_id"], ["prompt_templates.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("template_id", "version", name="uq_prompt_template_version"),
    )

    op.create_table(
        "playground_experiments",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("owner_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="active"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_playground_experiments_owner_id", "playground_experiments", ["owner_id"])

    op.create_table(
        "playground_experiment_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("experiment_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("owner_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("prompt_version_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("prompt_snapshot", sa.Text(), nullable=True),
        sa.Column("model_name", sa.String(128), nullable=True),
        sa.Column("generation_config", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("retrieval_scope", sa.String(32), nullable=False),
        sa.Column("document_ids", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("input_question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("citations", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("metrics", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("token_usage", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("estimated_cost_usd", sa.Float(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["experiment_id"], ["playground_experiments.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["prompt_version_id"], ["prompt_template_versions.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "document_promotion_audits",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("actor_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("previous_visibility", sa.String(16), nullable=False),
        sa.Column("new_visibility", sa.String(16), nullable=False),
        sa.Column("outcome", sa.String(32), nullable=False),
        sa.Column("request_id", sa.String(64), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_document_promotion_audits_document_id", "document_promotion_audits", ["document_id"]
    )

    template_rows = []
    version_rows = []
    for tid, name, desc, text in SYSTEM_TEMPLATES:
        vid = uuid.uuid4()
        template_rows.append(
            {
                "id": tid,
                "owner_id": None,
                "name": name,
                "description": desc,
                "current_version": 1,
                "is_system": True,
            }
        )
        version_rows.append(
            {
                "id": vid,
                "template_id": tid,
                "version": 1,
                "template_text": text,
                "config": {},
                "created_by": None,
            }
        )
    op.bulk_insert(
        sa.table(
            "prompt_templates",
            sa.column("id", postgresql.UUID),
            sa.column("owner_id", postgresql.UUID),
            sa.column("name", sa.String),
            sa.column("description", sa.Text),
            sa.column("current_version", sa.Integer),
            sa.column("is_system", sa.Boolean),
        ),
        template_rows,
    )
    op.bulk_insert(
        sa.table(
            "prompt_template_versions",
            sa.column("id", postgresql.UUID),
            sa.column("template_id", postgresql.UUID),
            sa.column("version", sa.Integer),
            sa.column("template_text", sa.Text),
            sa.column("config", postgresql.JSONB),
            sa.column("created_by", postgresql.UUID),
        ),
        version_rows,
    )


def downgrade() -> None:
    op.drop_index("ix_document_promotion_audits_document_id", "document_promotion_audits")
    op.drop_table("document_promotion_audits")
    op.drop_table("playground_experiment_runs")
    op.drop_index("ix_playground_experiments_owner_id", "playground_experiments")
    op.drop_table("playground_experiments")
    op.drop_table("prompt_template_versions")
    op.drop_index("ix_prompt_templates_owner_id", "prompt_templates")
    op.drop_table("prompt_templates")
