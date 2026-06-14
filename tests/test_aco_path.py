"""Tests para ACOConversationPath (Fase 4 - Módulo E)."""

import pytest
from unittest.mock import MagicMock

from src.conversation.aco_path import ACOConversationPath


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _Msg:
    """Mensaje de conversación simple para tests."""
    def __init__(self, text: str):
        self.text = text


@pytest.fixture
def aco():
    return ACOConversationPath(n_ants=5, max_iter=5, random_state=0)


def _make_predictor(score: float = 80.0, label: str = "fraudulent"):
    pred = MagicMock()
    pred.predict.return_value = {"risk_score": score, "predicted_class": label}
    return pred


FRAUD_MSGS = [
    _Msg("Hola, ¿cómo estás?"),
    _Msg("ALERTA BBVA: su cuenta fue bloqueada por actividad sospechosa."),
    _Msg("Verifique sus credenciales en http://bbva-seguro.com urgente."),
    _Msg("Proporcione su NIP para desbloquear su tarjeta hoy mismo."),
]


# ---------------------------------------------------------------------------
# Tests de estructura del resultado
# ---------------------------------------------------------------------------

def test_analyze_returns_expected_keys(aco):
    result = aco.analyze(FRAUD_MSGS)
    expected = {
        "worst_path", "worst_path_texts", "path_score",
        "escalation_start", "pheromone_map", "convergence", "manipulation_arc",
    }
    assert expected.issubset(result.keys())


def test_worst_path_valid_indices(aco):
    result = aco.analyze(FRAUD_MSGS)
    n = len(FRAUD_MSGS)
    for idx in result["worst_path"]:
        assert 0 <= idx < n


def test_path_score_in_range(aco):
    result = aco.analyze(FRAUD_MSGS)
    assert 0.0 <= result["path_score"] <= 1.0


def test_escalation_start_valid_index(aco):
    result = aco.analyze(FRAUD_MSGS)
    n = len(FRAUD_MSGS)
    assert 0 <= result["escalation_start"] < n


def test_convergence_length(aco):
    result = aco.analyze(FRAUD_MSGS)
    assert len(result["convergence"]) == aco.max_iter


def test_pheromone_map_dimensions(aco):
    result = aco.analyze(FRAUD_MSGS)
    n = len(FRAUD_MSGS)
    assert len(result["pheromone_map"]) == n
    for row in result["pheromone_map"]:
        assert len(row) == n


def test_worst_path_texts_match_path(aco):
    result = aco.analyze(FRAUD_MSGS)
    for i, idx in enumerate(result["worst_path"]):
        expected_text = FRAUD_MSGS[idx].text
        assert result["worst_path_texts"][i] == expected_text


def test_manipulation_arc_is_string(aco):
    result = aco.analyze(FRAUD_MSGS)
    assert isinstance(result["manipulation_arc"], str)
    assert len(result["manipulation_arc"]) > 0


def test_single_message_does_not_fail(aco):
    msgs = [_Msg("Solo un mensaje de prueba.")]
    result = aco.analyze(msgs)
    assert result["worst_path"] == [0]
    assert result["path_score"] >= 0.0


def test_empty_messages_returns_empty(aco):
    result = aco.analyze([])
    assert result["worst_path"] == []
    assert result["path_score"] == 0.0


def test_analyze_without_predictor_uses_heuristic(aco):
    """Sin predictor, debe usar heurística interna sin lanzar excepción."""
    result = aco.analyze(FRAUD_MSGS, predictor=None)
    assert "path_score" in result
    assert result["path_score"] >= 0.0
