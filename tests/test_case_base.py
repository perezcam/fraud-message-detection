"""Tests para src/detection/case_base.py"""

import numpy as np
import pandas as pd
import pytest

from src.detection.case_base import CaseBase
from src.ml.features import build_tfidf_vectorizer, fit_tfidf


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_df():
    msgs = [
        "Urgente: haga clic en el link para verificar su contraseña",
        "Su cuenta fue bloqueada. Confirme datos ahora",
        "ALERTA BBVA: verifique su tarjeta en http://fraude.com",
        "Hola, cómo estás? Nos vemos el viernes",
        "Recuerda la reunión de mañana a las 10am",
        "Tu pedido llegará el jueves. Gracias por tu compra",
        "Premio ganador, haga depósito para reclamar",
        "No ignores este aviso, su cuenta será cancelada",
    ]
    labels = ["fraudulent", "fraudulent", "fraudulent", "legitimate",
              "legitimate", "legitimate", "fraudulent", "fraudulent"]
    return pd.DataFrame({"message": msgs, "label": labels})


@pytest.fixture
def vectorizer_and_cb(small_df):
    texts = small_df["message"].tolist()
    vec, _ = fit_tfidf(texts, save=False)
    cb = CaseBase()
    cb.build(small_df, vec)
    return vec, cb


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCaseBase:
    def test_build_returns_counts(self, small_df):
        texts = small_df["message"].tolist()
        vec, _ = fit_tfidf(texts, save=False)
        cb = CaseBase()
        info = cb.build(small_df, vec)
        assert info["n_cases"] == len(small_df)
        assert info["n_fraud"] + info["n_legit"] == info["n_cases"]

    def test_query_returns_required_keys(self, vectorizer_and_cb):
        _, cb = vectorizer_and_cb
        result = cb.query("verifique su cuenta BBVA")
        for key in ("cbr_score", "top_k_labels", "similarities", "most_similar"):
            assert key in result

    def test_cbr_score_range(self, vectorizer_and_cb):
        _, cb = vectorizer_and_cb
        result = cb.query("su cuenta fue bloqueada urgente")
        assert 0.0 <= result["cbr_score"] <= 1.0

    def test_fraud_message_high_cbr(self, vectorizer_and_cb):
        _, cb = vectorizer_and_cb
        result = cb.query("Urgente: haga clic aquí para verificar su contraseña")
        assert result["cbr_score"] > 0.3

    def test_legit_message_low_cbr(self, vectorizer_and_cb):
        _, cb = vectorizer_and_cb
        result = cb.query("Hola, nos vemos el viernes en la reunión")
        assert result["cbr_score"] < 0.9

    def test_similarities_length(self, vectorizer_and_cb):
        _, cb = vectorizer_and_cb
        k = 3
        result = cb.query("test mensaje", k=k)
        assert len(result["top_k_labels"]) == k
        assert len(result["similarities"]) == k

    def test_most_similar_equals_max_sim(self, vectorizer_and_cb):
        _, cb = vectorizer_and_cb
        result = cb.query("cuenta bloqueada urgente")
        assert result["most_similar"] == max(result["similarities"])

    def test_query_before_build_raises(self):
        cb = CaseBase()
        with pytest.raises(RuntimeError):
            cb.query("test")

    def test_save_load(self, vectorizer_and_cb, tmp_path):
        vec, cb = vectorizer_and_cb
        path = tmp_path / "cb_test.npz"
        cb.save(path)

        cb2 = CaseBase()
        cb2.load(vec, path)
        r1 = cb.query("cuenta bloqueada urgente")
        r2 = cb2.query("cuenta bloqueada urgente")
        assert r1["cbr_score"] == r2["cbr_score"]

    def test_load_missing_file_raises(self, vectorizer_and_cb, tmp_path):
        vec, _ = vectorizer_and_cb
        cb = CaseBase()
        with pytest.raises(FileNotFoundError):
            cb.load(vec, tmp_path / "nonexistent.npz")
