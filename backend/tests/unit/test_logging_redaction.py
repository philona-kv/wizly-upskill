import logging

from app.observability.llm_usage import record_usage


def test_llm_usage_log_extra_has_no_prompt_fields(caplog) -> None:
    caplog.set_level(logging.INFO)
    record_usage(
        operation="generation",
        model="test-model",
        duration_ms=12,
        success=True,
        usage=None,
    )
    assert caplog.records
    payload = caplog.records[-1].__dict__
    forbidden = ("prompt", "content", "api_key", "sk-")
    for key, value in payload.items():
        if isinstance(value, str):
            lower = value.lower()
            assert not any(token in lower for token in forbidden)
