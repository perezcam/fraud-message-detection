"""Tests para los modelos ML ampliados (XGBoost, LightGBM, calibración)."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.calibration import CalibratedClassifierCV

from src.ml.train import get_model, train
from src.ml.features import fit_tfidf, extract_manual_features_batch, combine_features


@pytest.fixture
def small_df():
    """Dataset mínimo con las dos clases para pruebas rápidas."""
    texts_fraud = [
        "URGENT: click http://evil.com send your PIN now",
        "Your account is blocked verify password immediately",
        "You won $5000 send card number to claim prize",
        "Bank alert: provide OTP code or account suspended",
        "Transfer money now or your account will be closed",
    ] * 6  # 30 fraud
    texts_legit = [
        "Hi, your order is ready for pickup",
        "Meeting at 3pm tomorrow, see you there",
        "Thanks for your payment, receipt attached",
        "Reminder: doctor appointment on Friday",
        "Your package has been delivered successfully",
    ] * 6  # 30 legit
    return pd.DataFrame({
        "message": texts_fraud + texts_legit,
        "label": ["fraudulent"] * 30 + ["legitimate"] * 30,
    })


class TestGetModel:
    def test_xgboost_instantiates(self):
        pytest.importorskip("xgboost")
        model = get_model("xgboost")
        assert model is not None

    def test_lightgbm_instantiates(self):
        pytest.importorskip("lightgbm")
        model = get_model("lightgbm")
        assert model is not None

    def test_linear_svc_calibrated_instantiates(self):
        model = get_model("linear_svc_calibrated")
        assert isinstance(model, CalibratedClassifierCV)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="no soportado"):
            get_model("nonexistent_model")


class TestXGBoostTrain:
    def test_xgboost_train_and_predict(self, small_df):
        pytest.importorskip("xgboost")
        result = train(small_df, model_name="xgboost", save=False)
        assert "model" in result
        preds = result["model"].predict(result["X_test"])
        assert len(preds) == len(result["y_test"])

    def test_xgboost_has_predict_proba(self, small_df):
        pytest.importorskip("xgboost")
        result = train(small_df, model_name="xgboost", save=False)
        assert hasattr(result["model"], "predict_proba")
        proba = result["model"].predict_proba(result["X_test"])
        assert proba.shape == (len(result["y_test"]), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


class TestLightGBMTrain:
    def test_lightgbm_train_and_predict(self, small_df):
        pytest.importorskip("lightgbm")
        result = train(small_df, model_name="lightgbm", save=False)
        preds = result["model"].predict(result["X_test"])
        assert len(preds) == len(result["y_test"])

    def test_lightgbm_has_predict_proba(self, small_df):
        pytest.importorskip("lightgbm")
        result = train(small_df, model_name="lightgbm", save=False)
        proba = result["model"].predict_proba(result["X_test"])
        assert proba.shape[1] == 2


class TestLinearSVCCalibrated:
    def test_calibrated_has_predict_proba(self, small_df):
        result = train(small_df, model_name="linear_svc_calibrated", save=False)
        assert hasattr(result["model"], "predict_proba")
        proba = result["model"].predict_proba(result["X_test"])
        assert proba.shape[1] == 2
        # probabilidades bien calibradas: suma a 1
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


class TestSparseCompatibility:
    """Verifica que XGBoost y LightGBM aceptan matrices sparse directamente."""

    def _make_sparse_data(self, small_df):
        texts = small_df["message"].tolist()
        vec, X_tfidf = fit_tfidf(texts, save=False)
        X_manual = extract_manual_features_batch(texts)
        X = combine_features(X_tfidf, X_manual)
        y = (small_df["label"] == "fraudulent").astype(int).values
        return X, y

    def test_xgboost_accepts_sparse(self, small_df):
        pytest.importorskip("xgboost")
        X, y = self._make_sparse_data(small_df)
        assert sp.issparse(X)
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=5, eval_metric="logloss", random_state=42)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert len(preds) == len(y)

    def test_lightgbm_accepts_sparse(self, small_df):
        pytest.importorskip("lightgbm")
        X, y = self._make_sparse_data(small_df)
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(n_estimators=5, random_state=42, verbose=-1)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert len(preds) == len(y)
