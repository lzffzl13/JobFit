"""Unit tests for LLM extraction — JSON parsing + field fallback."""

import pytest

from app.services.llm import (
    _parse_json_strict,
    _safe_dict,
    _safe_float,
    _safe_list,
    _safe_str,
)


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


class TestParseJsonStrict:
    def test_valid_json(self):
        result = _parse_json_strict('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_in_code_block(self):
        text = 'Here is the result:\n```json\n{"key": "value"}\n```'
        result = _parse_json_strict(text)
        assert result == {"key": "value"}

    def test_json_in_code_block_no_lang(self):
        text = '```\n{"key": "value"}\n```'
        result = _parse_json_strict(text)
        assert result == {"key": "value"}

    def test_json_with_surrounding_text(self):
        text = 'The result is: {"key": "value"} hope this helps'
        result = _parse_json_strict(text)
        assert result == {"key": "value"}

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            _parse_json_strict("this is not json at all")

    def test_nested_json(self):
        text = '{"skills": {"hard": ["Python", "FastAPI"]}}'
        result = _parse_json_strict(text)
        assert result["skills"]["hard"] == ["Python", "FastAPI"]


# ---------------------------------------------------------------------------
# Safe type coercion
# ---------------------------------------------------------------------------


class TestSafeList:
    def test_none_returns_empty(self):
        assert _safe_list(None) == []

    def test_list_pass_through(self):
        assert _safe_list([1, 2, 3]) == [1, 2, 3]

    def test_scalar_wrapped(self):
        assert _safe_list("hello") == ["hello"]

    def test_dict_wrapped(self):
        result = _safe_list({"a": 1})
        assert result == [{"a": 1}]


class TestSafeStr:
    def test_none_returns_default(self):
        assert _safe_str(None) == ""

    def test_none_returns_custom_default(self):
        assert _safe_str(None, "N/A") == "N/A"

    def test_string_pass_through(self):
        assert _safe_str("hello") == "hello"

    def test_number_to_string(self):
        assert _safe_str(42) == "42"

    def test_bool_to_string(self):
        assert _safe_str(True) == "True"


class TestSafeFloat:
    def test_none_returns_zero(self):
        assert _safe_float(None) == 0.0

    def test_int_to_float(self):
        assert _safe_float(3) == 3.0

    def test_string_number(self):
        assert _safe_float("2.5") == 2.5

    def test_invalid_returns_default(self):
        assert _safe_float("abc") == 0.0

    def test_custom_default(self):
        assert _safe_float(None, -1.0) == -1.0


class TestSafeDict:
    def test_none_returns_empty(self):
        assert _safe_dict(None) == {}

    def test_dict_pass_through(self):
        assert _safe_dict({"a": 1}) == {"a": 1}

    def test_non_dict_returns_empty(self):
        assert _safe_dict("hello") == {}
        assert _safe_dict([1, 2]) == {}
