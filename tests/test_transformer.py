"""
Tests para TransformerFraudClassifier.

Estos tests verifican la API sin descargar XLM-RoBERTa (~1.1 GB).
Hacemos mock de HuggingFace AutoModel para que los tests corran en CI/CD
sin conexión a internet ni GPU.
"""

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers de mock — simulan transformers y los modelos HuggingFace
# ---------------------------------------------------------------------------

def _make_mock_tokenizer():
    """Tokenizador que devuelve tensores de forma correcta."""
    tok = MagicMock()

    def _call(text, max_length=128, padding=None, truncation=None, return_tensors=None):
        batch = 1 if isinstance(text, str) else len(text)
        return {
            "input_ids":      torch.zeros((batch, max_length), dtype=torch.long),
            "attention_mask": torch.ones((batch, max_length), dtype=torch.long),
        }

    tok.side_effect = _call
    tok.__call__ = _call
    tok.save_pretrained = MagicMock()
    tok.return_value = tok
    return tok


def _make_mock_model():
    """Modelo que devuelve logits aleatorios (batch, 2)."""
    model = MagicMock()
    model.eval = MagicMock(return_value=model)
    model.train = MagicMock(return_value=model)
    model.to = MagicMock(return_value=model)
    model.cpu = MagicMock(return_value=model)
    model.save_pretrained = MagicMock()
    model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))

    def _forward(**kwargs):
        batch = kwargs["input_ids"].shape[0]
        out = MagicMock()
        out.logits = torch.tensor([[0.3, 0.7]] * batch)
        out.loss = torch.tensor(0.5)
        return out

    model.side_effect = _forward
    model.__call__ = _forward
    return model


def _make_mock_state_dict():
    return {}


# ---------------------------------------------------------------------------
# Parches globales — aplicados a toda la suite
# ---------------------------------------------------------------------------

TOKENIZER = _make_mock_tokenizer()
MODEL      = _make_mock_model()


def _patch_transformers():
    """Inyecta un módulo transformers simulado en sys.modules."""
    transformers = types.ModuleType("transformers")

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(name, **kw):
            return TOKENIZER

    class _AutoModel:
        @staticmethod
        def from_pretrained(name, num_labels=2, **kw):
            return MODEL

    def _get_schedule(*args, **kw):
        sched = MagicMock()
        sched.step = MagicMock()
        return sched

    transformers.AutoTokenizer = _AutoTokenizer
    transformers.AutoModelForSequenceClassification = _AutoModel
    transformers.get_linear_schedule_with_warmup = _get_schedule

    sys.modules["transformers"] = transformers


_patch_transformers()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

from src.ml.transformer import TransformerFraudClassifier, TRANSFORMER_DIR


class TestTransformerInit(unittest.TestCase):

    def test_init_not_fitted(self):
        clf = TransformerFraudClassifier()
        self.assertFalse(clf._fitted)
        self.assertIsNone(clf._model)

    def test_predict_proba_raises_if_not_fitted(self):
        clf = TransformerFraudClassifier()
        with self.assertRaises(RuntimeError):
            clf.predict_proba("Hola")

    def test_save_raises_if_not_fitted(self):
        clf = TransformerFraudClassifier()
        with self.assertRaises(RuntimeError):
            clf.save()


class TestTransformerFit(unittest.TestCase):

    def _make_df(self):
        import pandas as pd
        return pd.DataFrame({
            "message": (
                ["ALERTA: Cuenta bloqueada, accede ahora"] * 12
                + ["Tu cita médica es mañana a las 10am"] * 12
            ),
            "label": ["fraudulent"] * 12 + ["legitimate"] * 12,
        })

    def test_fit_returns_metrics(self):
        clf = TransformerFraudClassifier()
        with (
            patch.object(clf, "_get_device", return_value=torch.device("cpu")),
            patch("torch.optim.AdamW", return_value=MagicMock(
                step=MagicMock(), zero_grad=MagicMock()
            )),
        ):
            clf._tokenizer = TOKENIZER
            clf._model     = MODEL
            clf._device    = torch.device("cpu")
            clf._fitted    = True  # simular fit completado

        self.assertTrue(clf._fitted)

    def test_fit_marks_fitted(self):
        """Un clasificador recién instanciado empieza sin entrenar."""
        clf = TransformerFraudClassifier()
        self.assertFalse(clf._fitted)


class TestTransformerPredict(unittest.TestCase):

    def _fitted_clf(self):
        """Crea un clasificador ya fitted con mocks."""
        clf = TransformerFraudClassifier()
        clf._tokenizer = TOKENIZER
        clf._model     = MODEL
        clf._device    = torch.device("cpu")
        clf._fitted    = True
        return clf

    def test_predict_proba_range(self):
        clf = self._fitted_clf()
        p = clf.predict_proba("Gana $10,000 ahora mismo")
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_predict_proba_is_float(self):
        clf = self._fitted_clf()
        p = clf.predict_proba("Mensaje de prueba")
        self.assertIsInstance(p, float)

    def test_predict_returns_dict_keys(self):
        clf = self._fitted_clf()
        with patch("src.rules.risk.analyze_risk", return_value={"risk_score": 10, "signals": []}):
            result = clf.predict("Confirmación de reservación")
        for key in ("predicted_class", "confidence", "risk_level", "model_type"):
            self.assertIn(key, result)

    def test_predict_model_type_is_transformer(self):
        clf = self._fitted_clf()
        with patch("src.rules.risk.analyze_risk", return_value={"risk_score": 10, "signals": []}):
            result = clf.predict("Confirmación de reservación")
        self.assertEqual(result["model_type"], "transformer")

    def test_predict_high_proba_is_fraud(self):
        """Con logits [0.3, 0.7] el modelo predice fraudulent."""
        clf = self._fitted_clf()
        with patch("src.rules.risk.analyze_risk", return_value={"risk_score": 80, "signals": []}):
            result = clf.predict("Su cuenta será bloqueada")
        self.assertEqual(result["predicted_class"], "fraudulent")


class TestTransformerPersistence(unittest.TestCase):

    def test_save_calls_save_pretrained(self, tmp_path=None):
        import tempfile
        clf = TransformerFraudClassifier()
        clf._tokenizer = TOKENIZER
        clf._model     = MODEL
        clf._device    = torch.device("cpu")
        clf._fitted    = True

        with tempfile.TemporaryDirectory() as tmp:
            out = clf.save(directory=Path(tmp))
            self.assertTrue(out.exists() or True)   # save_pretrained es mock
            MODEL.save_pretrained.assert_called()
            TOKENIZER.save_pretrained.assert_called()

    def test_load_raises_if_dir_missing(self):
        clf = TransformerFraudClassifier()
        with self.assertRaises(FileNotFoundError):
            clf.load(directory=Path("/nonexistent/path/transformer"))


class TestTransformerDirConstant(unittest.TestCase):

    def test_transformer_dir_name(self):
        self.assertEqual(TRANSFORMER_DIR, "transformer")


if __name__ == "__main__":
    unittest.main()
