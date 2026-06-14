"""Tests para el módulo de detección de anomalías (Isolation Forest)."""

import pandas as pd
import pytest

from src.ml.anomaly import AnomalyDetector


@pytest.fixture
def detector_fitted():
    df = pd.DataFrame({
        "message": [
            "Hi, your order is ready for pickup",
            "Reminder: meeting at 3pm tomorrow",
            "Thanks for your payment, receipt attached",
            "Your doctor appointment is on Friday at 10am",
            "Package delivered successfully to your address",
            "Please confirm your attendance for the event",
            "Your reservation has been confirmed",
            "Call us back when you have a moment",
            "The report is ready to be reviewed",
            "See you at the office tomorrow morning",
            "Lunch is at noon, don't forget",
            "Happy birthday! Wishing you all the best",
        ] * 4,
        "label": ["legitimate"] * 48,
    })
    det = AnomalyDetector()
    det.fit(df)
    return det


class TestAnomalyDetectorFit:
    def test_fit_requires_legitimate_messages(self):
        df = pd.DataFrame({
            "message": ["fraud msg"] * 3,
            "label": ["fraudulent"] * 3,
        })
        det = AnomalyDetector()
        with pytest.raises(ValueError, match="al menos 10 mensajes legítimos"):
            det.fit(df)

    def test_fit_sets_fitted_flag(self, detector_fitted):
        assert detector_fitted._fitted is True

    def test_fit_initializes_components(self, detector_fitted):
        assert detector_fitted._vectorizer is not None
        assert detector_fitted._model is not None


class TestAnomalyDetectorScore:
    def test_score_returns_float_in_range(self, detector_fitted):
        score = detector_fitted.score("Hello, your order is ready")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_legitimate_message_low_score(self, detector_fitted):
        score = detector_fitted.score("Your appointment is confirmed for tomorrow")
        assert score < 0.9

    def test_empty_message_returns_zero(self, detector_fitted):
        assert detector_fitted.score("") == 0.0
        assert detector_fitted.score("   ") == 0.0

    def test_score_before_fit_raises(self):
        det = AnomalyDetector()
        with pytest.raises(RuntimeError, match="no entrenado"):
            det.score("some message")


class TestAnomalyDetectorPersistence:
    def test_save_and_load(self, detector_fitted, tmp_path):
        path = tmp_path / "test_anomaly.joblib"
        detector_fitted.save(path)

        det2 = AnomalyDetector()
        det2.load(path)
        assert det2._fitted is True

        score_orig = detector_fitted.score("Your order is ready")
        score_load = det2.score("Your order is ready")
        assert abs(score_orig - score_load) < 1e-6

    def test_load_missing_file_raises(self):
        det = AnomalyDetector()
        with pytest.raises(FileNotFoundError):
            det.load("/nonexistent/path/anomaly.joblib")
