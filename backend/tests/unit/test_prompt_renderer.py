import pytest

from app.core.exceptions import ValidationFailedError
from app.services.prompt_renderer import render_prompt_template, validate_template_text


def test_render_allowed_variables() -> None:
    text = "Q: {question}\n{retrieved_context}"
    out = render_prompt_template(text, question="hi", retrieved_context="evidence")
    assert "hi" in out
    assert "evidence" in out


def test_reject_unknown_variable() -> None:
    with pytest.raises(ValidationFailedError):
        validate_template_text("Hello {evil}")


def test_missing_question_rejected() -> None:
    with pytest.raises(ValidationFailedError):
        validate_template_text("No vars here")
