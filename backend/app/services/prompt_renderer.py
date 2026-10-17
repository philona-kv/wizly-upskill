"""Safe prompt template rendering (no code execution)."""

from __future__ import annotations

import re
import string

from app.core.exceptions import ValidationFailedError

ALLOWED_VARIABLES = frozenset(
    {"question", "retrieved_context", "conversation_context", "citation_instructions"}
)
_PLACEHOLDER_RE = re.compile(r"\{([a-z_]+)\}")


def validate_template_text(template_text: str) -> None:
    for match in _PLACEHOLDER_RE.finditer(template_text):
        name = match.group(1)
        if name not in ALLOWED_VARIABLES:
            raise ValidationFailedError(
                f"Unknown template variable '{{{name}}}'. Allowed: {sorted(ALLOWED_VARIABLES)}.",
                code="invalid_template_variable",
            )
    for required in ("question",):
        if f"{{{required}}}" not in template_text:
            raise ValidationFailedError(
                f"Template must include {{{required}}}.",
                code="missing_template_variable",
            )


class _StrictFormatter(string.Formatter):
    def get_value(self, key, args, kwargs):
        if key not in ALLOWED_VARIABLES:
            raise ValidationFailedError(
                f"Unknown template variable '{key}'.", code="invalid_template_variable"
            )
        if key not in kwargs:
            raise ValidationFailedError(
                f"Missing value for template variable '{key}'.",
                code="missing_template_value",
            )
        return kwargs[key]


def render_prompt_template(template_text: str, **values: str) -> str:
    validate_template_text(template_text)
    extra = set(values) - ALLOWED_VARIABLES
    if extra:
        raise ValidationFailedError(
            f"Unexpected render keys: {sorted(extra)}", code="invalid_template_value"
        )
    filled = {k: values.get(k, "") for k in ALLOWED_VARIABLES}
    try:
        return _StrictFormatter().format(template_text, **filled)
    except KeyError as exc:
        raise ValidationFailedError(
            f"Invalid template placeholder: {exc}", code="invalid_template_syntax"
        ) from exc
