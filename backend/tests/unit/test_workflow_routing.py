from app.workflows.routing import after_assess, after_plan, after_validate


def test_after_plan_clarification() -> None:
    assert after_plan({"query_plan": {"needs_clarification": True}}) == "finalize_clarification"


def test_after_plan_retrieval() -> None:
    assert after_plan({"query_plan": {"needs_clarification": False}}) == "parallel_retrieval"


def test_after_assess_sufficient() -> None:
    assert after_assess({"evidence_assessment": {"sufficient": True}}) == "generate_answer"


def test_after_assess_retry() -> None:
    state = {
        "evidence_assessment": {"sufficient": False, "suggested_strategy": "rewrite_query"},
        "retrieval_attempt": 1,
        "max_retrieval_attempts": 2,
    }
    assert after_assess(state) == "plan_retrieval_retry"


def test_after_assess_abstain_when_no_docs() -> None:
    state = {
        "evidence_assessment": {"sufficient": False, "suggested_strategy": "none"},
        "retrieval_attempt": 2,
        "max_retrieval_attempts": 2,
        "document_evidence": [],
    }
    assert after_assess(state) == "finalize_abstain"


def test_after_assess_generate_when_docs_retrieved() -> None:
    state = {
        "evidence_assessment": {"sufficient": False, "suggested_strategy": "none"},
        "retrieval_attempt": 2,
        "max_retrieval_attempts": 2,
        "document_evidence": [{"source_type": "document", "content": "clinic setup steps"}],
    }
    assert after_assess(state) == "generate_answer"


def test_after_assess_retrieval_error() -> None:
    state = {
        "document_retrieval_error": "fail",
        "document_evidence": [],
        "selected_sources": ["documents"],
    }
    assert after_assess(state) == "finalize_error"


def test_after_validate_repair() -> None:
    state = {
        "validation_result": {"valid": False, "correction_required": True},
        "answer_repair_count": 0,
        "max_answer_repairs": 1,
    }
    assert after_validate(state) == "repair_answer"


def test_after_validate_success() -> None:
    assert after_validate({"validation_result": {"valid": True}}) == "finalize_success"
