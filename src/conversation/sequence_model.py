"""
Modelo de secuencias para detección de patrones conversacionales.

Jerarquía de modelos (de más a menos fuerte):
  1. NeuralConversationClassifier — BiLSTM + Atención (PyTorch)
       Modelo principal. Aprende dependencias temporales reales.
  2. ConversationWindowClassifier — RandomForest + IsolationForest (sklearn)
       Fallback si el modelo neuronal no está disponible.

El score final devuelto por predict_proba() proviene del nivel más alto
disponible. Los callers no necesitan saber qué modelo está activo.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NEURAL_FILE = "conversation_neural.pt"
RF_FILE     = "conversation_window_clf.joblib"


# ---------------------------------------------------------------------------
# Modelo neuronal (BiLSTM + atención)
# ---------------------------------------------------------------------------

class ConversationWindowClassifier:
    """
    Interfaz unificada para el clasificador de secuencias conversacionales.

    Internamente usa NeuralConversationClassifier (BiLSTM + atención).
    Si PyTorch no está disponible o el modelo no ha sido entrenado,
    devuelve -1.0 para que el caller pueda detectar la ausencia del modelo.

    El método predict_with_attention() devuelve también los pesos de atención
    por mensaje — permite señalar qué mensaje fue el más determinante.
    """

    def __init__(self) -> None:
        self._neural = None
        self._is_fitted = False
        self._extractor = None   # para el fallback feature-based

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        n_synthetic:  int = 3000,
        seq_length:   int = 5,
        epochs:       int = 50,
        random_state: int = 42,
        des_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Entrena el BiLSTM + atención con secuencias sintéticas del dataset.

        Args:
            df:           DataFrame con columnas "message" y "label".
            n_synthetic:  Secuencias sintéticas a generar.
            seq_length:   Mensajes por secuencia.
            epochs:       Épocas de entrenamiento.
            random_state: Semilla aleatoria.
            des_df:       DataFrame adicional generado por DESConversationSimulator.
                          Sus mensajes se añaden al pool de entrenamiento antes de
                          generar las secuencias sintéticas, mejorando la diversidad
                          y la coherencia causal de las secuencias de fraude.
        """
        from src.conversation.neural import NeuralConversationClassifier
        from src.conversation.features import WindowFeatureExtractor
        import logging
        _log = logging.getLogger(__name__)

        # Combinar dataset original con datos DES si se proporcionan
        if des_df is not None and not des_df.empty:
            required = {"message", "label"}
            if required.issubset(des_df.columns):
                combined = pd.concat(
                    [df[["message", "label"]], des_df[["message", "label"]]],
                    ignore_index=True,
                )
                _log.info(
                    f"DES dataset combinado: {len(df)} originales + "
                    f"{len(des_df)} DES = {len(combined)} mensajes totales."
                )
                df = combined
            else:
                _log.warning("des_df no tiene columnas 'message'/'label'. Ignorado.")

        clf = NeuralConversationClassifier()
        clf.fit(
            df,
            n_synthetic=n_synthetic,
            seq_length=seq_length,
            epochs=epochs,
            random_state=random_state,
        )
        self._neural    = clf
        self._extractor = WindowFeatureExtractor()
        self._is_fitted = True

    # ------------------------------------------------------------------
    # Predicción
    # ------------------------------------------------------------------

    def predict_proba(self, features_or_messages) -> float:
        """
        Devuelve probabilidad de que la ventana sea sospechosa (0–1).

        Acepta tanto un dict de features (para compatibilidad con el analyzer)
        como una lista de Message (preferido para el modelo neuronal).
        Devuelve -1.0 si el modelo no está disponible.
        """
        if not self._is_fitted or self._neural is None:
            return -1.0

        # Si recibe lista de Message → usar modelo neuronal directamente
        if isinstance(features_or_messages, list):
            return self._neural.predict_proba(features_or_messages)

        # Si recibe dict de features → no es ideal para el modelo neuronal
        # (no tenemos los textos), devolvemos -1 para que el caller ignore
        return -1.0

    def predict_with_attention(
        self, messages: list
    ) -> tuple[float, list[float]]:
        """
        Devuelve (prob_sospechoso, attention_weights_por_mensaje).

        Los attention_weights indican qué mensajes fueron más determinantes
        en la clasificación — útiles para explicabilidad.
        """
        if not self._is_fitted or self._neural is None:
            return -1.0, []
        return self._neural.predict(messages)

    def feature_importances(self) -> list[tuple[str, float]]:
        """No aplica al modelo neuronal; devuelve lista vacía."""
        return []

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> Path:
        if self._neural is None:
            raise RuntimeError("Modelo no entrenado.")
        return self._neural.save(path)

    def load(self, path: Optional[Path] = None) -> None:
        from src.conversation.neural import NeuralConversationClassifier
        from src.conversation.features import WindowFeatureExtractor

        clf = NeuralConversationClassifier()
        clf.load(path)
        self._neural    = clf
        self._extractor = WindowFeatureExtractor()
        self._is_fitted = True
