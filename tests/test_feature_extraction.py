"""Tests para el módulo de extracción de características."""

import numpy as np
import pytest
import scipy.sparse as sp

from src.ml.features import (
    MANUAL_FEATURE_NAMES,
    build_tfidf_vectorizer,
    combine_features,
    extract_manual_features,
    extract_manual_features_batch,
    fit_tfidf,
)

SAMPLE_TEXTS = [
    "click here https://spam.com win free money!!!",
    "hello how are you today friend",
    "your account has been blocked verify now urgente",
    "transferencia de $10,000 pesos a su cuenta bancaria",
]

# Vectorizador con min_df=1 para fixtures pequeñas (min_df=2 requiere corpus grande)
_TEST_VEC = build_tfidf_vectorizer(max_features=500, ngram_range=(1, 2))
_TEST_VEC.set_params(min_df=1)


class TestTfIdf:
    def test_fit_returns_correct_shape(self):
        _, X = fit_tfidf(SAMPLE_TEXTS, vectorizer=_TEST_VEC, save=False)
        assert X.shape[0] == len(SAMPLE_TEXTS)
        assert X.shape[1] > 0

    def test_transform_new_text(self):
        vec, _ = fit_tfidf(SAMPLE_TEXTS, vectorizer=_TEST_VEC, save=False)
        X_new = vec.transform(["win prize money"])
        assert X_new.shape[0] == 1

    def test_vectorizer_config(self):
        vec = build_tfidf_vectorizer(max_features=500, ngram_range=(1, 2))
        assert vec.max_features == 500
        assert vec.ngram_range == (1, 2)


class TestManualFeatures:
    def test_all_keys_present(self):
        features = extract_manual_features("Click https://evil.com to win $1000!")
        for key in MANUAL_FEATURE_NAMES:
            assert key in features, f"Falta característica: {key}"

    def test_url_detected(self):
        assert extract_manual_features("Visit https://evil.com now")["has_url"] == 1.0

    def test_no_url(self):
        assert extract_manual_features("Hello how are you")["has_url"] == 0.0

    def test_email_detected(self):
        assert extract_manual_features("Send to hacker@x.com")["has_email"] == 1.0

    def test_money_detected(self):
        assert extract_manual_features("Gana $5,000 pesos")["has_money"] == 1.0

    def test_exclamation_count(self):
        f = extract_manual_features("Act now!!!")
        assert f["exclamation_count"] == 3.0

    def test_question_count(self):
        f = extract_manual_features("¿Cómo estás? ¿Todo bien?")
        assert f["question_count"] == 2.0

    def test_empty_text_zero_features(self):
        f = extract_manual_features("")
        assert f["msg_length"] == 0.0
        assert f["word_count"] == 0.0
        assert f["has_url"] == 0.0

    def test_urgency_detected(self):
        f = extract_manual_features("Actúe urgente, cuenta bloqueada")
        assert f["has_urgency"] == 1.0

    def test_credential_words_detected(self):
        f = extract_manual_features("Envíe su contraseña y PIN ahora")
        assert f["has_credential_words"] == 1.0

    def test_prize_words_detected(self):
        f = extract_manual_features("Felicidades, usted es el ganador del sorteo")
        assert f["has_prize_words"] == 1.0


class TestBatchAndCombine:
    def test_batch_shape(self):
        result = extract_manual_features_batch(SAMPLE_TEXTS)
        assert result.shape == (len(SAMPLE_TEXTS), len(MANUAL_FEATURE_NAMES))

    def test_batch_dtype(self):
        result = extract_manual_features_batch(SAMPLE_TEXTS)
        assert result.dtype == float

    def test_combine_features_shape(self):
        vec, X_tfidf = fit_tfidf(SAMPLE_TEXTS, vectorizer=_TEST_VEC, save=False)
        manual = extract_manual_features_batch(SAMPLE_TEXTS)
        combined = combine_features(X_tfidf, manual)
        assert combined.shape[0] == len(SAMPLE_TEXTS)
        assert combined.shape[1] == X_tfidf.shape[1] + manual.shape[1]

    def test_combine_returns_sparse(self):
        vec, X_tfidf = fit_tfidf(SAMPLE_TEXTS, vectorizer=_TEST_VEC, save=False)
        manual = extract_manual_features_batch(SAMPLE_TEXTS)
        combined = combine_features(X_tfidf, manual)
        assert sp.issparse(combined)
