"""
Módulo de extracción de características.
Implementa TF-IDF y características manuales/estructurales combinables.
"""

import logging
import re
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    CREDENTIAL_WORDS,
    MODELS_DIR,
    MONEY_WORDS,
    NEGATION_WORDS,
    PRIZE_WORDS,
    TEXT_COLUMN,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
    URGENCY_WORDS,
    VECTORIZER_FILE_NAME,
)

logger = logging.getLogger(__name__)

_URL_RE   = re.compile(r"<URL>|https?://\S+|www\.\S+", re.IGNORECASE)
_PHONE_RE = re.compile(r"<PHONE>|\b\d{7,}\b")
_EMAIL_RE = re.compile(r"<EMAIL>|[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_MONEY_RE = re.compile(r"<MONEY>|\$|€|£|\b\d[\d,\.]*\s*(pesos?|dólares?|dollars?)\b", re.IGNORECASE)

MANUAL_FEATURE_NAMES = [
    "msg_length", "word_count", "exclamation_count", "question_count",
    "has_url", "has_phone", "has_email", "has_money",
    "uppercase_word_count", "has_urgency", "has_money_words",
    "has_credential_words", "has_prize_words",
    # Fase 3 — PLN
    "negation_score", "entity_count", "has_phone_ner", "has_amount_ner",
]

_WINDOW = 5  # tokens de ventana para detección de negación


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------

def build_tfidf_vectorizer(
    max_features: int = TFIDF_MAX_FEATURES,
    ngram_range: tuple[int, int] = TFIDF_NGRAM_RANGE,
    min_df: int = TFIDF_MIN_DF,
) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        strip_accents="unicode",
        analyzer="word",
        sublinear_tf=True,
    )


def fit_tfidf(
    texts: list[str],
    vectorizer: Optional[TfidfVectorizer] = None,
    save: bool = True,
) -> tuple[TfidfVectorizer, sp.csr_matrix]:
    if vectorizer is None:
        vectorizer = build_tfidf_vectorizer()
    X = vectorizer.fit_transform(texts)
    logger.info(f"TF-IDF ajustado: {X.shape[1]} características, {X.shape[0]} muestras.")
    if save:
        save_vectorizer(vectorizer)
    return vectorizer, X


def transform_tfidf(texts: list[str], vectorizer: TfidfVectorizer) -> sp.csr_matrix:
    return vectorizer.transform(texts)


def save_vectorizer(vectorizer: TfidfVectorizer, path: Optional[Path] = None) -> Path:
    if path is None:
        path = MODELS_DIR / VECTORIZER_FILE_NAME
    joblib.dump(vectorizer, path)
    logger.info(f"Vectorizador guardado en {path}")
    return path


def load_vectorizer(path: Optional[Path] = None) -> TfidfVectorizer:
    if path is None:
        path = MODELS_DIR / VECTORIZER_FILE_NAME
    vectorizer = joblib.load(path)
    logger.info(f"Vectorizador cargado desde {path}")
    return vectorizer


# ---------------------------------------------------------------------------
# Características manuales
# ---------------------------------------------------------------------------

def _has_word_list(text: str, word_list: list[str]) -> int:
    text_lower = text.lower()
    return int(any(w in text_lower for w in word_list))


def _negation_score(text: str) -> float:
    """
    Detecta si palabras de negación preceden a señales de fraude
    dentro de una ventana de _WINDOW tokens.

    Retorna un valor entre -1.0 y 0.0:
      -1.0 → negación clara ("no haga clic", "jamás envíes tu contraseña")
       0.0 → sin negación relevante detectada
    """
    tokens = text.lower().split()
    fraud_signals = set(URGENCY_WORDS + CREDENTIAL_WORDS + PRIZE_WORDS)
    score = 0.0
    for i, tok in enumerate(tokens):
        if tok in NEGATION_WORDS:
            window = tokens[i + 1: i + 1 + _WINDOW]
            if any(w in fraud_signals for w in window):
                score -= 1.0
    return max(score, -1.0)


def extract_manual_features(text: str) -> dict[str, float]:
    from src.ml.ner import extract_entities
    words = text.split()
    ents  = extract_entities(text)
    return {
        "msg_length":           float(len(text)),
        "word_count":           float(len(words)),
        "exclamation_count":    float(text.count("!")),
        "question_count":       float(text.count("?")),
        "has_url":              float(bool(_URL_RE.search(text))),
        "has_phone":            float(bool(_PHONE_RE.search(text))),
        "has_email":            float(bool(_EMAIL_RE.search(text))),
        "has_money":            float(bool(_MONEY_RE.search(text))),
        "uppercase_word_count": float(sum(1 for w in words if w.isupper() and len(w) > 1)),
        "has_urgency":          float(_has_word_list(text, URGENCY_WORDS)),
        "has_money_words":      float(_has_word_list(text, MONEY_WORDS)),
        "has_credential_words": float(_has_word_list(text, CREDENTIAL_WORDS)),
        "has_prize_words":      float(_has_word_list(text, PRIZE_WORDS)),
        # Fase 3 — PLN
        "negation_score":       _negation_score(text),
        "entity_count":         float(ents["entity_count"]),
        "has_phone_ner":        ents["has_phone"],
        "has_amount_ner":       ents["has_amount"],
    }


def extract_manual_features_batch(texts: Union[list[str], pd.Series]) -> np.ndarray:
    rows = [list(extract_manual_features(str(t)).values()) for t in texts]
    return np.array(rows, dtype=float)


def combine_features(tfidf_matrix: sp.csr_matrix, manual_matrix: np.ndarray) -> sp.csr_matrix:
    return sp.hstack([tfidf_matrix, sp.csr_matrix(manual_matrix)], format="csr")


def get_feature_names(vectorizer: TfidfVectorizer) -> list[str]:
    return vectorizer.get_feature_names_out().tolist() + MANUAL_FEATURE_NAMES
