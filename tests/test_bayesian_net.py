"""Tests para src/ml/bayesian_net.py"""

import numpy as np
import pandas as pd
import pytest

from src.ml.bayesian_net import FraudBayesNet, extract_bayes_features


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_df():
    msgs = [
        "Urgente: haga clic en http://fraude.com para verificar su contraseña BBVA",
        "Su cuenta HSBC fue suspendida. Llame ahora y proporcione su pin",
        "ALERTA: su tarjeta bloqueada, confirme datos en el link",
        "Hola, cómo estás? El fin de semana podemos vernos",
        "Recuerda llevar los documentos para la reunión de mañana",
        "Tu cita médica es a las 10am. Confirma asistencia",
        "Premio ganador: haga depósito de $500 ahora para reclamar su regalo",
        "No ignores este mensaje, tu cuenta fue limitada por Santander",
    ]
    labels = ["fraudulent", "fraudulent", "fraudulent", "legitimate",
              "legitimate", "legitimate", "fraudulent", "fraudulent"]
    return pd.DataFrame({"message": msgs, "label": labels})


@pytest.fixture
def trained_bn(small_df):
    bn = FraudBayesNet()
    bn.fit(small_df)
    return bn


# ---------------------------------------------------------------------------
# Tests extract_bayes_features
# ---------------------------------------------------------------------------

class TestExtractBayesFeatures:
    def test_returns_all_keys(self):
        feats = extract_bayes_features("test")
        expected = {"url_present", "urgency", "credential_request",
                    "bank_mentioned", "amount_present", "negation_present"}
        assert set(feats.keys()) == expected

    def test_url_detected(self):
        feats = extract_bayes_features("visite http://fraude.com")
        assert feats["url_present"] == 1

    def test_urgency_detected(self):
        feats = extract_bayes_features("Urgente: verifique su cuenta")
        assert feats["urgency"] == 1

    def test_bank_detected(self):
        feats = extract_bayes_features("Su cuenta BBVA fue bloqueada")
        assert feats["bank_mentioned"] == 1

    def test_negation_detected(self):
        feats = extract_bayes_features("no proporciones tu contraseña")
        assert feats["negation_present"] == 1

    def test_clean_message_all_zeros(self):
        feats = extract_bayes_features("Hola cómo estás buen día")
        assert all(v == 0 for v in feats.values())


# ---------------------------------------------------------------------------
# Tests FraudBayesNet
# ---------------------------------------------------------------------------

class TestFraudBayesNet:
    def test_fit_returns_metrics(self, small_df):
        bn = FraudBayesNet()
        metrics = bn.fit(small_df)
        assert "train_n" in metrics
        assert "auc_train" in metrics
        assert 0.0 <= metrics["auc_train"] <= 1.0

    def test_predict_proba_range(self, trained_bn):
        feats = {"url_present": 1, "urgency": 1, "credential_request": 1,
                 "bank_mentioned": 1, "amount_present": 0, "negation_present": 0}
        p = trained_bn.predict_proba(feats)
        assert 0.0 <= p <= 1.0

    def test_fraud_signal_higher_score(self, trained_bn):
        fraud_feats = {"url_present": 1, "urgency": 1, "credential_request": 1,
                       "bank_mentioned": 1, "amount_present": 1, "negation_present": 0}
        legit_feats = {"url_present": 0, "urgency": 0, "credential_request": 0,
                       "bank_mentioned": 0, "amount_present": 0, "negation_present": 0}
        assert trained_bn.predict_proba(fraud_feats) > trained_bn.predict_proba(legit_feats)

    def test_score_message(self, trained_bn):
        score = trained_bn.score_message(
            "Urgente: su cuenta BBVA fue bloqueada, verifique en http://fraude.com"
        )
        assert 0.0 <= score <= 1.0

    def test_predict_proba_unfitted_raises(self):
        bn = FraudBayesNet()
        with pytest.raises(RuntimeError):
            bn.predict_proba({"url_present": 0})

    def test_save_load(self, trained_bn, tmp_path):
        p = tmp_path / "bn_test.joblib"
        trained_bn.save(p)

        bn2 = FraudBayesNet()
        bn2.load(p)

        feats = {"url_present": 1, "urgency": 1, "credential_request": 1,
                 "bank_mentioned": 1, "amount_present": 0, "negation_present": 0}
        assert trained_bn.predict_proba(feats) == bn2.predict_proba(feats)

    def test_cpts_have_all_features(self, trained_bn):
        for feat in FraudBayesNet.FEATURES:
            assert feat in trained_bn._cpts
            assert 0 in trained_bn._cpts[feat]
            assert 1 in trained_bn._cpts[feat]
