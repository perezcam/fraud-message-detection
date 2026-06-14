"""Tests para el módulo de explicabilidad SHAP."""

import numpy as np
import pandas as pd
import pytest

from src.ml.explain import explain_prediction
from src.ml.train import train


@pytest.fixture(scope="module")
def trained_lr():
    df = pd.DataFrame({
        "message": (
            ["URGENT click http://evil.com send PIN now account blocked"] * 20
            + ["Your order is ready for pickup, see you soon"] * 20
        ),
        "label": ["fraudulent"] * 20 + ["legitimate"] * 20,
    })
    return train(df, model_name="logistic_regression", save=False)


@pytest.fixture(scope="module")
def trained_rf():
    pytest.importorskip("shap")
    df = pd.DataFrame({
        "message": (
            ["URGENT click http://evil.com send PIN now account blocked"] * 20
            + ["Your order is ready for pickup, see you soon"] * 20
        ),
        "label": ["fraudulent"] * 20 + ["legitimate"] * 20,
    })
    return train(df, model_name="random_forest", save=False)


class TestExplainPrediction:
    def test_returns_dict_with_required_keys(self, trained_lr):
        pytest.importorskip("shap")
        result = explain_prediction(
            "click here now http://evil.com send your password",
            trained_lr["model"],
            trained_lr["vectorizer"],
            trained_lr["int_to_label"],
            trained_lr["use_manual_features"],
        )
        assert "predicted_class" in result
        assert "top_features" in result
        assert "shap_available" in result

    def test_top_features_are_list(self, trained_lr):
        pytest.importorskip("shap")
        result = explain_prediction(
            "urgent verify your account now",
            trained_lr["model"],
            trained_lr["vectorizer"],
            trained_lr["int_to_label"],
            trained_lr["use_manual_features"],
        )
        if result["shap_available"]:
            assert isinstance(result["top_features"], list)
            for feat in result["top_features"]:
                assert "feature" in feat
                assert "shap_value" in feat
                assert feat["direction"] in ("fraud", "legit")

    def test_direction_matches_shap_sign(self, trained_lr):
        pytest.importorskip("shap")
        result = explain_prediction(
            "urgent verify your account now",
            trained_lr["model"],
            trained_lr["vectorizer"],
            trained_lr["int_to_label"],
            trained_lr["use_manual_features"],
        )
        if result["shap_available"]:
            for feat in result["top_features"]:
                if feat["shap_value"] > 0:
                    assert feat["direction"] == "fraud"
                else:
                    assert feat["direction"] == "legit"

    def test_works_without_shap(self, monkeypatch, trained_lr):
        """Debe retornar gracefully si shap no está instalado."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "shap":
                raise ImportError("shap not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        result = explain_prediction(
            "some message",
            trained_lr["model"],
            trained_lr["vectorizer"],
            trained_lr["int_to_label"],
            trained_lr["use_manual_features"],
        )
        assert result["shap_available"] is False

    def test_tree_explainer_for_random_forest(self, trained_rf):
        pytest.importorskip("shap")
        result = explain_prediction(
            "click here send your password immediately",
            trained_rf["model"],
            trained_rf["vectorizer"],
            trained_rf["int_to_label"],
            trained_rf["use_manual_features"],
        )
        assert "predicted_class" in result
        if result["shap_available"]:
            assert len(result["top_features"]) > 0
