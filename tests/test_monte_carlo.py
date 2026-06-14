"""Tests para MonteCarloAnalyzer (Fase 4 - Módulo C)."""

import pytest
from unittest.mock import MagicMock

from src.ml.monte_carlo import MonteCarloAnalyzer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def analyzer():
    return MonteCarloAnalyzer(n_simulations=20, random_state=0)


def _make_predictor(score: float = 70.0, label: str = "fraudulent"):
    """Predictor mock que siempre devuelve el mismo score."""
    pred = MagicMock()
    pred.predict.return_value = {"risk_score": score, "predicted_class": label}
    return pred


FRAUD_MSG   = "ALERTA BBVA: su cuenta fue bloqueada. Verifique en http://bbva-mx.com su NIP."
LEGIT_MSG   = "Hola, nos vemos mañana a las 10 para la junta."


# ---------------------------------------------------------------------------
# Tests de estructura del resultado
# ---------------------------------------------------------------------------

def test_analyze_returns_expected_keys(analyzer):
    pred = _make_predictor(80.0, "fraudulent")
    result = analyzer.analyze(FRAUD_MSG, pred)
    expected = {
        "mean_score", "std_score", "ci_low", "ci_high",
        "stability", "fraud_rate", "original_score", "n_simulations", "verdict",
    }
    assert expected.issubset(result.keys())


def test_n_simulations_in_result(analyzer):
    pred = _make_predictor()
    result = analyzer.analyze(FRAUD_MSG, pred)
    assert result["n_simulations"] == 20


def test_mean_score_range(analyzer):
    pred = _make_predictor(60.0, "fraudulent")
    result = analyzer.analyze(FRAUD_MSG, pred)
    assert 0 <= result["mean_score"] <= 100


def test_stability_range(analyzer):
    pred = _make_predictor(80.0, "fraudulent")
    result = analyzer.analyze(FRAUD_MSG, pred)
    assert 0.0 <= result["stability"] <= 1.0


def test_fraud_rate_range(analyzer):
    pred = _make_predictor(80.0, "fraudulent")
    result = analyzer.analyze(FRAUD_MSG, pred)
    assert 0.0 <= result["fraud_rate"] <= 1.0


def test_fraud_rate_high_for_fraud_predictor(analyzer):
    pred = _make_predictor(90.0, "fraudulent")
    result = analyzer.analyze(FRAUD_MSG, pred)
    assert result["fraud_rate"] == pytest.approx(1.0, abs=0.01)


def test_fraud_rate_zero_for_legit_predictor(analyzer):
    pred = _make_predictor(5.0, "legitimate")
    result = analyzer.analyze(LEGIT_MSG, pred)
    assert result["fraud_rate"] == pytest.approx(0.0, abs=0.01)


def test_ci_low_le_mean_le_ci_high(analyzer):
    pred = _make_predictor(50.0, "fraudulent")
    result = analyzer.analyze(FRAUD_MSG, pred)
    assert result["ci_low"] <= result["mean_score"] + 1e-6
    assert result["mean_score"] <= result["ci_high"] + 1e-6


def test_verdict_is_string(analyzer):
    pred = _make_predictor()
    result = analyzer.analyze(FRAUD_MSG, pred)
    assert isinstance(result["verdict"], str)
    assert len(result["verdict"]) > 0


def test_single_simulation_does_not_fail():
    mc = MonteCarloAnalyzer(n_simulations=1, random_state=7)
    pred = _make_predictor(70.0, "fraudulent")
    result = mc.analyze(FRAUD_MSG, pred)
    assert result["n_simulations"] == 1
    assert "mean_score" in result
