"""
Módulo de predicción.
Carga modelos entrenados y clasifica mensajes nuevos con salida estructurada.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import scipy.sparse as sp

from src.config import (
    METADATA_FILE_NAME,
    MODEL_FILE_TEMPLATE,
    MODELS_DIR,
    VECTORIZER_FILE_NAME,
)
from src.ml.features import (
    combine_features,
    extract_manual_features_batch,
    load_vectorizer,
)
from src.data.preprocessing import clean_text
from src.rules.risk import analyze_risk

logger = logging.getLogger(__name__)


def load_model(model_name: str, directory: Optional[Path] = None) -> Any:
    if directory is None:
        directory = MODELS_DIR
    path = directory / MODEL_FILE_TEMPLATE.format(model_name=model_name)
    if not path.exists():
        raise FileNotFoundError(f"Modelo no encontrado en: {path}")
    model = joblib.load(path)
    logger.info(f"Modelo cargado desde {path}")
    return model


def load_metadata(directory: Optional[Path] = None) -> dict:
    if directory is None:
        directory = MODELS_DIR
    path = directory / METADATA_FILE_NAME
    if not path.exists():
        raise FileNotFoundError(f"Metadatos no encontrados en: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _risk_level_from_label(label: str, confidence: Optional[float]) -> str:
    if label == "fraudulent":
        return "medium" if (confidence is not None and confidence < 0.6) else "high"
    if label == "suspicious":
        return "medium"
    return "medium" if (confidence is not None and confidence < 0.65) else "low"


def _get_recommendation(risk_level: str) -> str:
    return {
        "low":    "No se detectan señales fuertes de fraude.",
        "medium": "El mensaje contiene señales sospechosas. Verifique la fuente antes de responder.",
        "high":   "Posible fraude. No comparta datos personales, códigos ni realice pagos.",
    }.get(risk_level, "Verifique el mensaje con precaución.")


def _vectorize(clean_message: str, vectorizer: Any, use_manual_features: bool) -> sp.csr_matrix:
    X_tfidf = vectorizer.transform([clean_message])
    if use_manual_features:
        X_manual = extract_manual_features_batch([clean_message])
        return combine_features(X_tfidf, X_manual)
    return X_tfidf


def predict_message(
    message: str,
    model: Any,
    vectorizer: Any,
    int_to_label: dict,
    use_manual_features: bool = True,
) -> dict:
    clean = clean_text(message)
    risk_info = analyze_risk(message)

    if not clean:
        return {
            "original_message": message,
            "preprocessed_message": clean,
            "predicted_class": "legitimate",
            "risk_level": "low",
            "confidence": None,
            "risk_score": risk_info["risk_score"],
            "signals": risk_info["signals"],
            "recommendation": _get_recommendation("low"),
        }

    X = _vectorize(clean, vectorizer, use_manual_features)
    pred_int = int(model.predict(X)[0])
    predicted_label = int_to_label.get(pred_int, "unknown")

    confidence: Optional[float] = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        confidence = float(proba[pred_int])
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)[0]
        confidence = float(np.max(np.abs(scores)))

    model_risk = _risk_level_from_label(predicted_label, confidence)
    rule_risk  = risk_info["risk_level"]
    hierarchy  = {"low": 0, "medium": 1, "high": 2}
    final_risk = (
        model_risk
        if hierarchy.get(model_risk, 0) >= hierarchy.get(rule_risk, 0)
        else rule_risk
    )

    return {
        "original_message":     message,
        "preprocessed_message": clean,
        "predicted_class":      predicted_label,
        "risk_level":           final_risk,
        "confidence":           round(confidence, 4) if confidence is not None else None,
        "risk_score":           risk_info["risk_score"],
        "signals":              risk_info["signals"],
        "recommendation":       _get_recommendation(final_risk),
    }


def predict_batch(
    messages: list[str],
    model: Any,
    vectorizer: Any,
    int_to_label: dict,
    use_manual_features: bool = True,
) -> list[dict]:
    return [
        predict_message(msg, model, vectorizer, int_to_label, use_manual_features)
        for msg in messages
    ]


class FraudPredictor:
    """Wrapper de conveniencia que carga los artefactos guardados."""

    def __init__(self) -> None:
        self.metadata = load_metadata()
        self.model = load_model(self.metadata["model_name"])
        self.vectorizer = load_vectorizer()
        self.int_to_label: dict[int, str] = {
            int(k): v for k, v in self.metadata["int_to_label"].items()
        }
        self.use_manual_features: bool = self.metadata.get("use_manual_features", True)

    def predict(self, message: str) -> dict:
        return predict_message(
            message, self.model, self.vectorizer,
            self.int_to_label, self.use_manual_features,
        )

    def predict_batch(self, messages: list[str]) -> list[dict]:
        return predict_batch(
            messages, self.model, self.vectorizer,
            self.int_to_label, self.use_manual_features,
        )
