"""Tests para TabuOptimizer (Fase 4 - Módulo D)."""

import json
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from src.ml.tabu_optimizer import TabuOptimizer, TABU_THRESHOLDS_FILE
from src.ml.threshold_optimizer import DEFAULT_THRESHOLDS, _BOUNDS
from src.config import TEXT_COLUMN, LABEL_COLUMN


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_df():
    rows = []
    fraud_msgs = [
        "ALERTA BBVA bloqueada NIP urgente http://bbva-mx.com",
        "Su cuenta fue suspendida verifique credenciales hoy",
        "Deposite $500 para recuperar acceso bancario urgente",
        "HSBC tarjeta bloqueada confirme contraseña en enlace",
        "SAT irregularidades regularice tramite urgente NIP",
        "Paquete retenido deposite $200 liberar envío urgente",
    ]
    legit_msgs = [
        "Hola nos vemos mañana para la junta",
        "El reporte mensual quedó listo avísame",
        "Pueden confirmar la reunión del lunes",
        "El proveedor envió la factura por correo",
    ]
    for m in fraud_msgs:
        rows.append({TEXT_COLUMN: m, LABEL_COLUMN: "fraudulent"})
    for m in legit_msgs:
        rows.append({TEXT_COLUMN: m, LABEL_COLUMN: "legitimate"})
    return pd.DataFrame(rows)


def _make_cascade_mock():
    """Cascade stub que devuelve siempre predicción de fraude."""
    cascade = MagicMock()
    cascade.predict.return_value = {
        "risk_score": 80,
        "predicted_class": "fraudulent",
        "risk_level": "high",
    }
    return cascade


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_optimize_returns_expected_keys(small_df):
    opt = TabuOptimizer(max_iter=3, tabu_tenure=2, n_neighbors=2, random_state=0)
    cascade = _make_cascade_mock()
    with patch.object(opt._evaluator, "_evaluate", return_value=0.80):
        result = opt.optimize(cascade, small_df)
    expected = {"best_thresholds", "best_f1", "initial_f1", "improvement",
                "history", "n_tabu_moves"}
    assert expected.issubset(result.keys())


def test_best_thresholds_within_bounds(small_df):
    opt = TabuOptimizer(max_iter=3, tabu_tenure=2, n_neighbors=2, random_state=0)
    cascade = _make_cascade_mock()
    with patch.object(opt._evaluator, "_evaluate", return_value=0.80):
        result = opt.optimize(cascade, small_df)
    for key, (lo, hi) in _BOUNDS.items():
        val = result["best_thresholds"][key]
        assert lo - 0.1 <= val <= hi + 0.1, f"{key}={val} fuera de [{lo},{hi}]"


def test_n_tabu_moves_non_negative(small_df):
    opt = TabuOptimizer(max_iter=5, tabu_tenure=2, n_neighbors=3, random_state=0)
    cascade = _make_cascade_mock()
    with patch.object(opt._evaluator, "_evaluate", return_value=0.75):
        result = opt.optimize(cascade, small_df)
    assert result["n_tabu_moves"] >= 0


def test_history_length(small_df):
    opt = TabuOptimizer(max_iter=4, tabu_tenure=2, n_neighbors=2, random_state=0)
    cascade = _make_cascade_mock()
    with patch.object(opt._evaluator, "_evaluate", return_value=0.70):
        result = opt.optimize(cascade, small_df)
    # history tiene max_iter + 1 elementos (inicial + 1 por iter)
    assert len(result["history"]) == 5


def test_to_key_deterministic():
    opt = TabuOptimizer()
    k1 = opt._to_key(DEFAULT_THRESHOLDS)
    k2 = opt._to_key(DEFAULT_THRESHOLDS)
    assert k1 == k2
    assert isinstance(k1, tuple)


def test_tenure_one_does_not_fail(small_df):
    opt = TabuOptimizer(max_iter=3, tabu_tenure=1, n_neighbors=2, random_state=0)
    cascade = _make_cascade_mock()
    with patch.object(opt._evaluator, "_evaluate", return_value=0.65):
        result = opt.optimize(cascade, small_df)
    assert "best_thresholds" in result


def test_save_creates_json(tmp_path):
    opt = TabuOptimizer()
    out = tmp_path / "tabu_test.json"
    opt.save(DEFAULT_THRESHOLDS, path=out)
    assert out.exists()
    with open(out) as f:
        data = json.load(f)
    assert "gate1_conf" in data


def test_load_returns_default_when_missing():
    import pathlib, tempfile
    p = pathlib.Path(tempfile.mktemp(suffix=".json"))
    loaded = TabuOptimizer.load(path=p)
    assert loaded == DEFAULT_THRESHOLDS


def test_load_after_save(tmp_path):
    opt = TabuOptimizer()
    out = tmp_path / "tabu_roundtrip.json"
    opt.save(DEFAULT_THRESHOLDS, path=out)
    loaded = TabuOptimizer.load(path=out)
    assert set(loaded.keys()) == set(DEFAULT_THRESHOLDS.keys())
