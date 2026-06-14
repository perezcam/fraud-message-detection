"""Tests para EDAFraudGenerator (Fase 4 - Módulo B)."""

import pytest
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.ml.eda_fraud_generator import EDAFraudGenerator
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
        "SAT irregularidades regularice tramite-sat.mx urgente",
        "Paquete retenido deposite $200 para liberar envío",
    ]
    legit_msgs = [
        "Hola nos vemos mañana para la junta",
        "El reporte mensual quedó listo avísame",
        "Pueden confirmar la reunión del lunes",
    ]
    for m in fraud_msgs:
        rows.append({TEXT_COLUMN: m, LABEL_COLUMN: "fraudulent"})
    for m in legit_msgs:
        rows.append({TEXT_COLUMN: m, LABEL_COLUMN: "legitimate"})
    return pd.DataFrame(rows)


@pytest.fixture
def fitted_gen(small_df):
    vec = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    vec.fit(small_df[TEXT_COLUMN].tolist())
    gen = EDAFraudGenerator(random_state=0)
    gen.fit(small_df, vec)
    return gen


# ---------------------------------------------------------------------------
# Tests de fit()
# ---------------------------------------------------------------------------

def test_fit_returns_expected_keys(small_df):
    vec = TfidfVectorizer(max_features=500)
    vec.fit(small_df[TEXT_COLUMN].tolist())
    gen = EDAFraudGenerator(random_state=0)
    info = gen.fit(small_df, vec)
    assert {"n_fraud", "n_features", "top_fraud_words"}.issubset(info.keys())


def test_fit_n_fraud_count(small_df):
    vec = TfidfVectorizer(max_features=500)
    vec.fit(small_df[TEXT_COLUMN].tolist())
    gen = EDAFraudGenerator(random_state=0)
    info = gen.fit(small_df, vec)
    assert info["n_fraud"] == 6


def test_fit_raises_without_fraud():
    df = pd.DataFrame({TEXT_COLUMN: ["hola", "mundo"], LABEL_COLUMN: ["legitimate", "legitimate"]})
    vec = TfidfVectorizer()
    vec.fit(df[TEXT_COLUMN].tolist())
    gen = EDAFraudGenerator()
    with pytest.raises(ValueError):
        gen.fit(df, vec)


# ---------------------------------------------------------------------------
# Tests de generate_texts()
# ---------------------------------------------------------------------------

def test_generate_texts_count(fitted_gen):
    texts = fitted_gen.generate_texts(n=10)
    assert len(texts) == 10


def test_generate_texts_are_strings(fitted_gen):
    texts = fitted_gen.generate_texts(n=5)
    for t in texts:
        assert isinstance(t, str)
        assert len(t.strip()) > 0


def test_generate_texts_varied(fitted_gen):
    texts = fitted_gen.generate_texts(n=20)
    unique = set(texts)
    # Con 12 templates y vocabulario variable, al menos 5 distintos en 20
    assert len(unique) >= 5


def test_generate_texts_raises_before_fit():
    gen = EDAFraudGenerator()
    with pytest.raises(RuntimeError):
        gen.generate_texts(n=5)


# ---------------------------------------------------------------------------
# Tests de generate_dataframe()
# ---------------------------------------------------------------------------

def test_generate_dataframe_rows(fitted_gen):
    df = fitted_gen.generate_dataframe(n=15)
    assert len(df) == 15


def test_generate_dataframe_label_column(fitted_gen):
    df = fitted_gen.generate_dataframe(n=10)
    assert LABEL_COLUMN in df.columns
    assert (df[LABEL_COLUMN] == "fraudulent").all()


def test_generate_dataframe_text_column(fitted_gen):
    df = fitted_gen.generate_dataframe(n=10)
    assert TEXT_COLUMN in df.columns
    assert df[TEXT_COLUMN].str.len().min() > 0


# ---------------------------------------------------------------------------
# Tests de save/load
# ---------------------------------------------------------------------------

def test_save_and_load(tmp_path, fitted_gen):
    out = tmp_path / "eda_test.joblib"
    fitted_gen.save(path=out)
    assert out.exists()

    gen2 = EDAFraudGenerator(random_state=0)
    gen2.load(path=out)
    texts = gen2.generate_texts(n=3)
    assert len(texts) == 3


def test_save_raises_before_fit(tmp_path):
    gen = EDAFraudGenerator()
    with pytest.raises(RuntimeError):
        gen.save(path=tmp_path / "nope.joblib")
