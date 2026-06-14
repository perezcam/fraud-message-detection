"""
Razonamiento Basado en Casos (CBR) para detección de fraude.

Fuente: Sistemas Expertos (INFOS/7- Sistemas Expertos.pdf), sección Case-Based Reasoning.

El sistema "recuerda" todos los mensajes de entrenamiento. Para un mensaje nuevo,
encuentra los K más similares por cosine similarity sobre TF-IDF.
Si la mayoría son fraude → el nuevo probablemente también lo es.

Ventaja sobre el clasificador supervisado:
  - Captura patrones nuevos sin reentrenar
  - Funciona bien para variantes de fraudes ya vistos
  - La similitud es interpretable ("este mensaje se parece al fraude X")

Guardado en models/case_base.npz (matrix TF-IDF + etiquetas)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    CASE_BASE_FILE,
    LABEL_COLUMN,
    MODELS_DIR,
    TEXT_COLUMN,
)
from src.data.preprocessing import clean_text

logger = logging.getLogger(__name__)

_DEFAULT_K = 7


class CaseBase:
    """
    Base de casos para razonamiento CBR.

    Uso:
        cb = CaseBase()
        cb.build(df, vectorizer)
        cb.save()

        # En inferencia:
        cb.load(vectorizer)
        result = cb.query("Su cuenta fue bloqueada...")
        # → {"cbr_score": 0.86, "top_k_labels": [...], "similarities": [...]}
    """

    def __init__(self) -> None:
        self._matrix: Optional[sp.csr_matrix] = None
        self._labels: Optional[np.ndarray]    = None
        self._vectorizer = None
        self._built = False

    # ------------------------------------------------------------------
    # Construcción
    # ------------------------------------------------------------------

    def build(self, df, vectorizer) -> dict:
        """
        Construye la base de casos desde el dataset de entrenamiento.

        Args:
            df:         DataFrame con columnas TEXT_COLUMN y LABEL_COLUMN.
            vectorizer: TF-IDF vectorizador ya ajustado.

        Returns:
            dict con n_cases y distribución de etiquetas.
        """
        df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
        df = df[df[LABEL_COLUMN].isin(["fraudulent", "legitimate"])].copy()

        texts = df[TEXT_COLUMN].astype(str).apply(clean_text).tolist()
        self._matrix    = vectorizer.transform(texts)
        self._labels    = (df[LABEL_COLUMN] == "fraudulent").astype(int).values
        self._vectorizer = vectorizer
        self._built      = True

        n_fraud = int(self._labels.sum())
        n_legit = len(self._labels) - n_fraud
        logger.info(
            f"CaseBase construida: {len(texts)} casos "
            f"({n_fraud} fraude / {n_legit} legítimo)"
        )
        return {"n_cases": len(texts), "n_fraud": n_fraud, "n_legit": n_legit}

    # ------------------------------------------------------------------
    # Consulta
    # ------------------------------------------------------------------

    def query(self, message: str, k: int = _DEFAULT_K) -> dict:
        """
        Busca los K casos más similares y calcula el cbr_score.

        Args:
            message: Texto del mensaje a evaluar.
            k:       Número de casos vecinos.

        Returns:
            {
                "cbr_score":      float (0-1) — fracción de vecinos que son fraude,
                "top_k_labels":   list[int],
                "similarities":   list[float],
                "most_similar":   float — mayor similitud encontrada,
            }
        """
        if not self._built:
            raise RuntimeError("CaseBase no construida.")

        clean = clean_text(message) or message
        x_vec = self._vectorizer.transform([clean])

        sims  = cosine_similarity(x_vec, self._matrix)[0]
        top_k_idx = np.argsort(sims)[::-1][:k]

        top_labels = self._labels[top_k_idx].tolist()
        top_sims   = sims[top_k_idx].tolist()

        cbr_score = float(sum(top_labels) / max(k, 1))
        return {
            "cbr_score":    round(cbr_score, 4),
            "top_k_labels": top_labels,
            "similarities": [round(s, 4) for s in top_sims],
            "most_similar": round(float(max(top_sims)), 4),
        }

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> Path:
        if not self._built:
            raise RuntimeError("CaseBase no construida.")
        out = Path(path) if path else MODELS_DIR / CASE_BASE_FILE
        sp.save_npz(str(out), self._matrix)
        np.save(str(out).replace(".npz", "_labels.npy"), self._labels)
        logger.info(f"CaseBase guardada en {out}")
        return out

    def load(self, vectorizer, path: Optional[Path] = None) -> None:
        """Carga la base de casos desde disco. Requiere el vectorizador."""
        base = Path(path) if path else MODELS_DIR / CASE_BASE_FILE
        if not base.exists():
            raise FileNotFoundError(f"CaseBase no encontrada: {base}")
        labels_path = Path(str(base).replace(".npz", "_labels.npy"))
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels de CaseBase no encontradas: {labels_path}")

        self._matrix     = sp.load_npz(str(base))
        self._labels     = np.load(str(labels_path))
        self._vectorizer = vectorizer
        self._built      = True
        logger.info(
            f"CaseBase cargada: {len(self._labels)} casos desde {base}"
        )
