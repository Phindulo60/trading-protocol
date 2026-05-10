"""Tests for LLM trading analyst (mocked Bedrock)."""
import json
import pytest
from unittest.mock import patch, MagicMock

from fsp.llm.validator import SignalValidator, ValidationResult, SYSTEM_PROMPT


def _mock_response(decision, confidence, reason, analysis="Test analysis"):
    return {
        "output": {
            "message": {
                "content": [{"text": json.dumps({
                    "decision": decision,
                    "confidence": confidence,
                    "reason": reason,
                    "analysis": analysis,
                    "suggested_tp": None,
                    "suggested_sl": None,
                })}]
            }
        }
    }


@patch("fsp.llm.validator.boto3")
def test_validate_take(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.converse.return_value = _mock_response(
        "TAKE", 0.85, "Strong trend alignment with London session",
        "H1 bullish, LO session peak vol, no events. Good setup."
    )
    v = SignalValidator()
    result = v.validate("EURUSD", "long", "TREND_RSI", 1.17500, 1.17200, 1.18550)
    assert result.decision == "TAKE"
    assert result.confidence == 0.85
    assert result.analysis  # has analysis text


@patch("fsp.llm.validator.boto3")
def test_validate_skip(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.converse.return_value = _mock_response(
        "SKIP", 0.92, "NFP in 45 minutes",
        "US Non-Farm Payrolls release in 45 min. Never enter before NFP."
    )
    v = SignalValidator()
    result = v.validate("EURUSD", "short", "ARB", 1.17500, 1.17800, 1.16800)
    assert result.decision == "SKIP"


@patch("fsp.llm.validator.boto3")
def test_validate_reduce(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.converse.return_value = _mock_response(
        "REDUCE", 0.7, "Correlated exposure with existing GBPUSD long",
        "Already long GBPUSD. Adding EURUSD long doubles USD-short risk."
    )
    v = SignalValidator()
    result = v.validate("EURUSD", "long", "LEVEL_OB", 1.17500, 1.17200, 1.18250,
                        context={"signals_today": [{"pair": "GBPUSD", "direction": "long"}]})
    assert result.decision == "REDUCE"


@patch("fsp.llm.validator.boto3")
def test_fallback_to_haiku(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    # First call (Sonnet) fails, second (Haiku) succeeds
    mock_client.converse.side_effect = [
        Exception("Model unavailable"),
        _mock_response("TAKE", 0.6, "Haiku fallback", "Sonnet down, basic check OK"),
    ]
    v = SignalValidator()
    result = v.validate("EURUSD", "long", "TREND_RSI", 1.17500, 1.17200, 1.18550)
    assert result.decision == "TAKE"
    assert mock_client.converse.call_count == 2


@patch("fsp.llm.validator.boto3")
def test_both_models_fail(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.converse.side_effect = Exception("Network timeout")
    v = SignalValidator()
    result = v.validate("EURUSD", "long", "TREND_RSI", 1.17500, 1.17200, 1.18550)
    assert result.decision == "TAKE"  # safe default
    assert result.confidence == 0.5


@patch("fsp.llm.validator.boto3")
def test_suggested_levels(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.converse.return_value = {
        "output": {"message": {"content": [{"text": json.dumps({
            "decision": "TAKE",
            "confidence": 0.8,
            "reason": "Good setup but tighten SL",
            "analysis": "ATR suggests tighter stop is optimal.",
            "suggested_tp": 1.18800,
            "suggested_sl": 1.17300,
        })}]}}
    }
    v = SignalValidator()
    result = v.validate("EURUSD", "long", "ARB", 1.17500, 1.17200, 1.18550)
    assert result.suggested_tp == 1.18800
    assert result.suggested_sl == 1.17300


def test_system_prompt_is_analyst():
    """Ensure prompt positions LLM as analyst, not filter."""
    assert "co-pilot" in SYSTEM_PROMPT or "analyst" in SYSTEM_PROMPT
    assert "decisive" in SYSTEM_PROMPT.lower() or "decision" in SYSTEM_PROMPT.lower()
    assert "analysis" in SYSTEM_PROMPT.lower()
