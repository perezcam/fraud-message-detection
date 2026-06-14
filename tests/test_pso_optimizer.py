"""Tests para PSOOptimizer (Fase 4 - Módulo A)."""

import json
import pytest
import pandas as pd

from src.ml.pso_optimizer import PSOOptimizer, DEFAULT_PARAMS, _BOUNDS
from src.config import TEXT_COLUMN, LABEL_COLUMN


# ---------------------------------------------------------------------------
# Fixture de dataset mínimo
# ---------------------------------------------------------------------------

@pytest.fixture
def small_df():
    rows = []
    fraud_msgs = [
        "ALERTA BBVA bloqueada NIP urgente http://bbva-mx.com",
        "Su cuenta fue suspendida verifique credenciales hoy",
        "Deposite $500 para recuperar acceso bancario urgente",
        "HSBC: tarjeta bloqueada, confirme contraseña en enlace",
        "SAT detectó irregularidades regularice en tramite-sat.mx",
        "Paquete retenido deposite $200 para liberar envío",
        "Citibanamex: OTP requerido acceso no autorizado detectado",
        "Verificacion urgente acceso sospechoso cuenta SAT NIP",
    ]
    legit_msgs = [
        "Hola, nos vemos mañana para la junta.",
        "El reporte mensual quedó listo, avísame.",
        "Pueden confirmar la reunión del lunes?",
        "El proveedor envió la factura por correo.",
        "Recuerda traer la presentación mañana.",
        "Quedamos de vernos a las 3pm en la oficina.",
        "El sistema estará en mantenimiento el sábado.",
        "Por favor envía el contrato firmado esta semana.",
    ]
    for m in fraud_msgs:
        rows.append({TEXT_COLUMN: m, LABEL_COLUMN: "fraudulent"})
    for m in legit_msgs:
        rows.append({TEXT_COLUMN: m, LABEL_COLUMN: "legitimate"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_optimize_returns_expected_keys(small_df):
    opt = PSOOptimizer(n_particles=2, max_iter=2, random_state=0)
    result = opt.optimize(small_df)
    expected = {"best_params", "best_f1", "initial_f1", "improvement",
                "convergence", "n_evaluations"}
    assert expected.issubset(result.keys())


def test_best_f1_in_range(small_df):
    opt = PSOOptimizer(n_particles=2, max_iter=2, random_state=0)
    result = opt.optimize(small_df)
    assert 0.0 <= result["best_f1"] <= 1.0


def test_convergence_length(small_df):
    opt = PSOOptimizer(n_particles=2, max_iter=5, random_state=0)
    result = opt.optimize(small_df)
    # convergence tiene max_iter + 1 elementos (inicial + 1 por iteración)
    assert len(result["convergence"]) == 6


def test_best_params_within_bounds(small_df):
    opt = PSOOptimizer(n_particles=2, max_iter=2, random_state=0)
    result = opt.optimize(small_df)
    for key, (lo, hi) in _BOUNDS.items():
        val = result["best_params"][key]
        assert lo - 1 <= val <= hi + 1, f"{key}={val} fuera de [{lo},{hi}]"


def test_n_evaluations_positive(small_df):
    opt = PSOOptimizer(n_particles=2, max_iter=2, random_state=0)
    result = opt.optimize(small_df)
    assert result["n_evaluations"] >= 2


def test_single_particle_does_not_fail(small_df):
    opt = PSOOptimizer(n_particles=1, max_iter=1, random_state=0)
    result = opt.optimize(small_df)
    assert "best_params" in result


def test_save_creates_json(tmp_path, small_df):
    opt = PSOOptimizer(n_particles=2, max_iter=2, random_state=0)
    result = opt.optimize(small_df)
    out = tmp_path / "pso_test.json"
    path = opt.save(result["best_params"], path=out)
    assert path.exists()
    with open(path) as f:
        data = json.load(f)
    assert "n_estimators" in data


def test_load_returns_default_when_missing():
    import tempfile, pathlib
    p = pathlib.Path(tempfile.mktemp(suffix=".json"))
    loaded = PSOOptimizer.load(path=p)
    assert loaded == DEFAULT_PARAMS


def test_load_after_save(tmp_path, small_df):
    opt = PSOOptimizer(n_particles=2, max_iter=2, random_state=0)
    result = opt.optimize(small_df)
    out = tmp_path / "pso_roundtrip.json"
    opt.save(result["best_params"], path=out)
    loaded = PSOOptimizer.load(path=out)
    assert set(loaded.keys()) == set(DEFAULT_PARAMS.keys())


def test_improvement_is_float(small_df):
    opt = PSOOptimizer(n_particles=2, max_iter=2, random_state=0)
    result = opt.optimize(small_df)
    assert isinstance(result["improvement"], float)
