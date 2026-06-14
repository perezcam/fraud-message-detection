"""
Detección de anomalías para mensajes fraudulentos no vistos.

Entrena un Isolation Forest SOLO con mensajes legítimos.
Un mensaje muy distante de esa distribución recibe un anomaly_score alto,
incluso si el clasificador ML lo marca como legítimo.

Esto detecta patrones de fraude nuevos (zero-day) que no estaban en entrenamiento.
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import LABEL_COLUMN, MODELS_DIR, RANDOM_STATE, TEXT_COLUMN
from src.data.preprocessing import clean_text

logger = logging.getLogger(__name__)

ANOMALY_FILE       = "anomaly_detector.joblib"
_TFIDF_MAX         = 3000   # vocabulario reducido para el detector de anomalías
_IF_CONTAMINATION  = 0.05   # proporción esperada de anomalías en el set legítimo
_ANOMALY_THRESHOLD = 0.75   # score ≥ threshold → elevar riesgo


class AnomalyDetector:
    """
    Detector de anomalías basado en Isolation Forest.

    Flujo:
        1. fit(df)    — entrena sobre mensajes legítimos únicamente
        2. save()     — persiste en models/anomaly_detector.joblib
        3. load()     — carga el modelo en memoria
        4. score(msg) — retorna anomaly_score ∈ [0, 1]
                        (0 = muy similar a legítimos, 1 = muy anómalo)

    Integración en cascade.py:
        - Si anomaly_score ≥ _ANOMALY_THRESHOLD y ML dice "legitimate"
          → elevar a risk_level "medium"
        - El score se incluye siempre en el resultado final del CascadePredictor
    """

    def __init__(self) -> None:
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._model: Optional[IsolationForest] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> None:
        legit = df[df[LABEL_COLUMN] == "legitimate"][TEXT_COLUMN].dropna()
        if len(legit) < 10:
            raise ValueError(
                f"Se necesitan al menos 10 mensajes legítimos para entrenar el AnomalyDetector. "
                f"Encontrados: {len(legit)}"
            )

        texts = legit.astype(str).apply(clean_text).tolist()
        logger.info(f"Entrenando AnomalyDetector con {len(texts)} mensajes legítimos.")

        self._vectorizer = TfidfVectorizer(
            max_features=_TFIDF_MAX,
            ngram_range=(1, 2),
            min_df=2,
            strip_accents="unicode",
            sublinear_tf=True,
        )
        X = self._vectorizer.fit_transform(texts)

        self._model = IsolationForest(
            n_estimators=200,
            contamination=_IF_CONTAMINATION,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        self._model.fit(X)
        self._fitted = True
        logger.info("AnomalyDetector entrenado correctamente.")

    # ------------------------------------------------------------------
    # Inferencia
    # ------------------------------------------------------------------

    def score(self, message: str) -> float:
        """
        Retorna anomaly_score ∈ [0, 1].

        0   → muy similar a mensajes legítimos (no anómalo)
        1   → muy alejado de la distribución legítima (muy anómalo)

        Usa la función decision_function del Isolation Forest, que retorna
        valores negativos para anomalías y positivos para inliers.
        Se normaliza al rango [0, 1] donde 1 = anómalo.
        """
        if not self._fitted:
            raise RuntimeError("AnomalyDetector no entrenado. Llama a fit() o load() primero.")

        clean = clean_text(message)
        if not clean:
            return 0.0

        X = self._vectorizer.transform([clean])
        # decision_function: valores negativos = anómalo, positivos = normal
        raw = float(self._model.decision_function(X)[0])

        # Convertir a [0, 1]: cuanto más negativo, más anómalo → score más alto
        # Rango empírico típico: [-0.5, 0.5] → clamp y normalizar
        clamped = max(-0.5, min(0.5, raw))
        anomaly_score = round((0.5 - clamped) / 1.0, 4)
        return anomaly_score

    def is_anomalous(self, message: str) -> bool:
        return self.score(message) >= _ANOMALY_THRESHOLD

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> Path:
        if not self._fitted:
            raise RuntimeError("Modelo no entrenado.")
        out = Path(path) if path else MODELS_DIR / ANOMALY_FILE
        joblib.dump({"vectorizer": self._vectorizer, "model": self._model}, out)
        logger.info(f"AnomalyDetector guardado en {out}")
        return out

    def load(self, path: Optional[Path] = None) -> None:
        p = Path(path) if path else MODELS_DIR / ANOMALY_FILE
        if not p.exists():
            raise FileNotFoundError(f"AnomalyDetector no encontrado: {p}")
        data = joblib.load(p)
        self._vectorizer = data["vectorizer"]
        self._model      = data["model"]
        self._fitted     = True
        logger.info(f"AnomalyDetector cargado desde {p}")
