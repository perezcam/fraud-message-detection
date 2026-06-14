"""
Módulo de evaluación de modelos.
Calcula métricas, genera reportes y guarda la matriz de confusión.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import FIGURES_DIR, METRICS_DIR

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[list[str]] = None,
) -> dict:
    """
    Calcula métricas de clasificación.
    El recall de la clase 'fraudulent' se reporta por separado dado su
    importancia crítica en este dominio.
    """
    n_classes = len(np.unique(y_true))
    average = "binary" if n_classes == 2 else "weighted"

    metrics: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "classification_report": classification_report(
            y_true, y_pred, target_names=target_names, zero_division=0
        ),
    }

    if target_names:
        for i, name in enumerate(target_names):
            mask = y_true == i
            if mask.sum() > 0:
                metrics[f"recall_{name}"] = float(
                    recall_score(y_true == i, y_pred == i, zero_division=0)
                )
                metrics[f"precision_{name}"] = float(
                    precision_score(y_true == i, y_pred == i, zero_division=0)
                )

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    model_name: str = "model",
    save_dir: Optional[Path] = None,
) -> Path:
    """Genera y guarda la matriz de confusión como imagen PNG."""
    if save_dir is None:
        save_dir = FIGURES_DIR

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=f"Matriz de Confusión — {model_name}",
        ylabel="Etiqueta real",
        xlabel="Etiqueta predicha",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    out_path = save_dir / f"confusion_matrix_{model_name}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Matriz de confusión guardada en {out_path}")
    return out_path


def save_metrics_report(
    metrics: dict,
    model_name: str = "model",
    save_dir: Optional[Path] = None,
) -> Path:
    """Guarda métricas numéricas en JSON y el reporte de texto en TXT."""
    if save_dir is None:
        save_dir = METRICS_DIR

    numeric_metrics = {k: v for k, v in metrics.items() if k != "classification_report"}
    json_path = save_dir / f"metrics_{model_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(numeric_metrics, f, indent=2)

    txt_path = save_dir / f"classification_report_{model_name}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Modelo: {model_name}\n\n")
        f.write(metrics.get("classification_report", ""))

    logger.info(f"Métricas guardadas en {json_path}")
    return json_path


def evaluate(
    model: Any,
    X_test,
    y_test: np.ndarray,
    int_to_label: dict,
    model_name: str = "model",
    vectorizer: Any = None,
    use_manual_features: bool = True,
) -> dict:
    """
    Pipeline completo de evaluación: predicción, métricas, reportes y gráficas.
    Enfatiza el recall de la clase 'fraudulent' como métrica crítica.
    Si se provee vectorizer, genera también el SHAP summary plot.
    """
    y_pred = model.predict(X_test)
    class_names = [int_to_label[i] for i in sorted(int_to_label.keys())]

    metrics = compute_metrics(y_test, y_pred, target_names=class_names)

    logger.info(f"\n{metrics['classification_report']}")

    if "recall_fraudulent" in metrics:
        logger.info(
            f"[CRÍTICO] Recall clase 'fraudulent': {metrics['recall_fraudulent']:.4f} "
            f"(cuanto más alto, menos fraudes sin detectar)"
        )

    plot_confusion_matrix(y_test, y_pred, class_names, model_name)
    save_metrics_report(metrics, model_name)

    # SHAP summary plot (cuando el vectorizador está disponible)
    if vectorizer is not None:
        try:
            from src.ml.explain import plot_shap_summary
            shap_path = plot_shap_summary(
                model=model,
                X_sample=X_test,
                vectorizer=vectorizer,
                use_manual_features=use_manual_features,
                model_name=model_name,
            )
            if shap_path:
                metrics["shap_summary_path"] = str(shap_path)
        except Exception as exc:
            logger.warning(f"SHAP summary plot omitido: {exc}")

    return metrics
