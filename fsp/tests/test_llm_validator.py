"""Tests for LLM signal validator (mocked Bedrock)."""
import json
import pytest
from unittest.mock import patch, MagicMock

from fsp.llm.validator import SignalValidator, ValidationResult, SYSTEM_PROMPT


@patch("fsp.llm.validator.boto3")
def test_validate_take(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.converse.return_value = {
        "output": {
            "message": {
                "content": [{"text": json.dumps({
                    "decision": "TAKE",
                    "confidence": 0.85,
                    "reason": "Strong trend alignment with London session"
                })}]
            }
        }
    }

    v = SignalValidator()
    result = v.validate("EURUSD", "long", "TREND_RSI", 1.17500, 1.17200, 1.18550)

    assert result.decision == "TAKE"
    assert result.confidence == 0.85
    assert "trend" in result.reason.lower()


@patch("fsp.llm.validator.boto3")
def test_validate_skip(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.converse.return_value = {
        "output": {
            "message": {
                "content": [{"text": json.dumps({
                    "decision": "SKIP",
                    "confidence": 0.92,
                    "reason": "NFP release in 45 minutes"
                })}]
            }
        }
    }

    v = SignalValidator()
    result = v.validate("EURUSD", "short", "ARB", 1.17500, 1.17800, 1.16800)

    assert result.decision == "SKIP"
    assert result.confidence == 0.92


@patch("fsp.llm.validator.boto3")
def test_validate_reduce(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.converse.return_value = {
        "output": {
            "message": {
                "content": [{"text": json.dumps({
                    "decision": "REDUCE",
                    "confidence": 0.7,
                    "reason": "Already long GBPUSD, correlated exposure"
                })}]
            }
        }
    }

    v = SignalValidator()
    result = v.validate(
        "EURUSD", "long", "LEVEL_OB", 1.17500, 1.17200, 1.18250,
        context={"open_positions": [{"pair": "GBPUSD", "direction": "long"}]}
    )

    assert result.decision == "REDUCE"


@patch("fsp.llm.validator.boto3")
def test_fallback_on_error(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.converse.side_effect = Exception("Network timeout")

    v = SignalValidator()
    result = v.validate("EURUSD", "long", "TREND_RSI", 1.17500, 1.17200, 1.18550)

    # Should default to TAKE on failure
    assert result.decision == "TAKE"
    assert result.confidence == 0.5
    assert "unavailable" in result.reason.lower()


@patch("fsp.llm.validator.boto3")
def test_malformed_json_fallback(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.converse.return_value = {
        "output": {
            "message": {
                "content": [{"text": "I think you should take this trade because..."}]
            }
        }
    }

    v = SignalValidator()
    result = v.validate("EURUSD", "long", "ARB", 1.17500, 1.17200, 1.18550)

    # Non-JSON response defaults to TAKE
    assert result.decision == "TAKE"


def test_system_prompt_has_rules():
    """Ensure key decision rules are in the system prompt."""
    assert "NFP" in SYSTEM_PROMPT or "high-impact" in SYSTEM_PROMPT
    assert "SKIP" in SYSTEM_PROMPT
    assert "REDUCE" in SYSTEM_PROMPT
    assert "JSON" in SYSTEM_PROMPT
