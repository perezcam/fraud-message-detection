"""
Módulo de entrenamiento.
Soporta múltiples modelos supervisados y guarda los artefactos del experimento.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.config import (
    DEFAULT_MODEL,
    LABEL_COLUMN,
    METADATA_FILE_NAME,
    MODEL_FILE_TEMPLATE,
    MODELS_DIR,
    RANDOM_STATE,
    SUPPORTED_MODELS,
    TEST_SIZE,
    TEXT_COLUMN,
    VECTORIZER_FILE_NAME,
)
from src.ml.features import (
    combine_features,
    extract_manual_features_batch,
    fit_tfidf,
    save_vectorizer,
)
from src.data.preprocessing import preprocess_dataframe

logger = logging.getLogger(__name__)


def get_model(model_name: str, **kwargs) -> Any:
    key = model_name.lower().replace("-", "_").replace(" ", "_")

    if key not in SUPPORTED_MODELS:
        raise ValueError(
            f"Modelo no soportado: '{model_name}'. Opciones válidas: {SUPPORTED_MODELS}"
        )

    if key == "naive_bayes":
        return MultinomialNB(**kwargs)
    if key == "logistic_regression":
        return LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced", **kwargs
        )
    if key == "linear_svc":
        return LinearSVC(
            max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced", **kwargs
        )
    if key == "linear_svc_calibrated":
        return CalibratedClassifierCV(
            LinearSVC(max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced"),
            cv=5,
        )
    if key == "random_forest":
        return RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced", **kwargs
        )
    if key == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("pip install xgboost") from exc
        return XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1, **kwargs,
        )
    if key == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError("pip install lightgbm") from exc
        return LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, **kwargs,
        )


def save_model(model: Any, model_name: str, directory: Optional[Path] = None) -> Path:
    if directory is None:
        directory = MODELS_DIR
    path = directory / MODEL_FILE_TEMPLATE.format(model_name=model_name)
    joblib.dump(model, path)
    logger.info(f"Modelo guardado en {path}")
    return path


def save_metadata(metadata: dict, directory: Optional[Path] = None) -> Path:
    if directory is None:
        directory = MODELS_DIR
    path = directory / METADATA_FILE_NAME
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Metadatos guardados en {path}")
    return path


def encode_labels(labels: pd.Series) -> tuple[np.ndarray, dict[str, int]]:
    unique = sorted(labels.unique())
    label_to_int = {lbl: idx for idx, lbl in enumerate(unique)}
    return labels.map(label_to_int).values, label_to_int


def _build_features(
    X_train_text: list[str],
    X_test_text: list[str],
    model_name: str,
    use_manual_features: bool,
    save: bool,
) -> tuple[sp.csr_matrix, sp.csr_matrix, Any]:
    vectorizer, X_train_tfidf = fit_tfidf(X_train_text, save=save)
    X_test_tfidf = vectorizer.transform(X_test_text)

    if use_manual_features and model_name != "naive_bayes":
        X_train_manual = extract_manual_features_batch(X_train_text)
        X_test_manual  = extract_manual_features_batch(X_test_text)
        X_train = combine_features(X_train_tfidf, X_train_manual)
        X_test  = combine_features(X_test_tfidf,  X_test_manual)
    else:
        if use_manual_features and model_name == "naive_bayes":
            logger.warning("MultinomialNB no soporta características manuales. Se usará solo TF-IDF.")
        X_train = X_train_tfidf
        X_test  = X_test_tfidf

    return X_train, X_test, vectorizer


def train(
    df: pd.DataFrame,
    model_name: str = DEFAULT_MODEL,
    use_manual_features: bool = True,
    save: bool = True,
    use_pso_params: bool = False,
) -> dict:
    logger.info(f"Iniciando entrenamiento con modelo: {model_name}")

    df = preprocess_dataframe(df)
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])

    X_text = df[TEXT_COLUMN].tolist()
    y, label_map = encode_labels(df[LABEL_COLUMN])
    int_to_label = {v: k for k, v in label_map.items()}

    logger.info(f"Mapa de etiquetas: {label_map}")
    unique, counts = np.unique(y, return_counts=True)
    logger.info(f"Distribución: { {int_to_label[u]: int(c) for u, c in zip(unique, counts)} }")

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_test, vectorizer = _build_features(
        X_train_text, X_test_text, model_name, use_manual_features, save
    )

    # Usar hiperparámetros optimizados por PSO si están disponibles y se solicitaron
    pso_kwargs: dict = {}
    if use_pso_params and model_name == "lightgbm":
        try:
            from src.ml.pso_optimizer import PSOOptimizer
            pso_kwargs = PSOOptimizer.load()
            logger.info(f"Hiperparámetros PSO cargados: {pso_kwargs}")
        except Exception as exc:
            logger.warning(f"No se pudieron cargar hiperparámetros PSO: {exc}")

    model = get_model(model_name, **pso_kwargs)
    model.fit(X_train, y_train)
    logger.info("Entrenamiento completado.")

    if save:
        save_model(model, model_name)
        metadata = {
            "model_name": model_name,
            "label_map": label_map,
            "int_to_label": int_to_label,
            "train_samples": len(X_train_text),
            "test_samples": len(X_test_text),
            "use_manual_features": use_manual_features and model_name != "naive_bayes",
            "features_count": X_train.shape[1],
            "classes": list(label_map.keys()),
        }
        save_metadata(metadata)

    return {
        "model": model, "vectorizer": vectorizer,
        "label_map": label_map, "int_to_label": int_to_label,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "X_train_text": X_train_text, "X_test_text": X_test_text,
        "use_manual_features": use_manual_features and model_name != "naive_bayes",
    }
