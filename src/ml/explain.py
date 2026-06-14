"""
Explicabilidad de predicciones con SHAP.

Soporta:
  - TreeExplainer  → RandomForest, XGBoost, LightGBM  (rápido, exacto)
  - LinearExplainer → LogisticRegression, LinearSVC calibrado

Retorna las N features más importantes para una predicción individual,
indicando si empujaron hacia "fraudulent" o hacia "legitimate".
También genera el beeswarm plot global del conjunto de evaluación.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)

_TREE_MODELS  = ("RandomForestClassifier", "XGBClassifier", "LGBMClassifier")
_LINEAR_MODELS = ("LogisticRegression", "CalibratedClassifierCV", "LinearSVC")
_TOP_N = 10


def _model_type(model: Any) -> str:
    return type(model).__name__


def _get_feature_names(vectorizer: Any, use_manual_features: bool) -> list[str]:
    from src.ml.features import MANUAL_FEATURE_NAMES
    names = vectorizer.get_feature_names_out().tolist()
    if use_manual_features:
        names += MANUAL_FEATURE_NAMES
    return names


def _to_dense(X: Any) -> np.ndarray:
    return X.toarray() if sp.issparse(X) else np.asarray(X)


# ---------------------------------------------------------------------------
# Explicación de una muestra individual
# ---------------------------------------------------------------------------

def explain_prediction(
    message: str,
    model: Any,
    vectorizer: Any,
    int_to_label: dict,
    use_manual_features: bool = True,
    top_n: int = _TOP_N,
) -> dict:
    """
    Retorna las features más influyentes para la predicción de un mensaje.

    Returns:
        {
            "predicted_class": str,
            "top_features": [{"feature": str, "shap_value": float, "direction": "fraud"|"legit"}],
            "base_value": float,
            "shap_available": bool,
        }
    """
    try:
        import shap
    except ImportError:
        return {"shap_available": False, "top_features": [], "error": "pip install shap"}

    from src.data.preprocessing import clean_text
    from src.ml.features import combine_features, extract_manual_features_batch, load_vectorizer

    clean = clean_text(message)
    X_tfidf = vectorizer.transform([clean])
    if use_manual_features:
        X_manual = extract_manual_features_batch([clean])
        X = combine_features(X_tfidf, X_manual)
    else:
        X = X_tfidf

    pred_int = int(model.predict(X)[0])
    predicted_class = int_to_label.get(pred_int, "unknown")

    feature_names = _get_feature_names(vectorizer, use_manual_features)
    X_dense = _to_dense(X)

    try:
        mname = _model_type(model)
        fraud_idx = next(
            (k for k, v in int_to_label.items() if v == "fraudulent"), 1
        )

        if any(t in mname for t in _TREE_MODELS):
            explainer = shap.TreeExplainer(model)
            shap_out  = explainer.shap_values(X_dense)

            # Normalizar salida: puede ser Explanation, lista de arrays o array 3D
            if hasattr(shap_out, "values"):
                # Objeto Explanation (SHAP >= 0.40)
                vals = shap_out.values  # (n, features) o (n, features, n_classes)
                if vals.ndim == 3:
                    vals = vals[:, :, fraud_idx]
                sv = vals[0]
                ev = shap_out.base_values
                base_val = float(ev[0, fraud_idx] if ev.ndim == 2 else ev[0])
            elif isinstance(shap_out, list):
                sv_list = shap_out
                sv = sv_list[fraud_idx][0] if fraud_idx < len(sv_list) else sv_list[-1][0]
                ev = explainer.expected_value
                if hasattr(ev, "__len__"):
                    base_val = float(ev[fraud_idx] if fraud_idx < len(ev) else ev[-1])
                else:
                    base_val = float(ev)
            else:
                # array 3D: (n, features, n_classes)
                sv = shap_out[0, :, fraud_idx] if shap_out.ndim == 3 else shap_out[0]
                ev = explainer.expected_value
                if hasattr(ev, "__len__"):
                    base_val = float(ev[fraud_idx] if fraud_idx < len(ev) else ev[-1])
                else:
                    base_val = float(ev)
        else:
            # LinearExplainer necesita datos de fondo — usamos zeros
            background = np.zeros((1, X_dense.shape[1]))
            explainer  = shap.LinearExplainer(model, background)
            shap_out   = explainer.shap_values(X_dense)

            if isinstance(shap_out, list):
                sv = shap_out[fraud_idx][0] if fraud_idx < len(shap_out) else shap_out[-1][0]
            elif hasattr(shap_out, "values"):
                vals = shap_out.values
                sv   = vals[0, :, fraud_idx] if vals.ndim == 3 else vals[0]
            else:
                sv = shap_out[0]

            ev = explainer.expected_value
            if hasattr(ev, "__len__"):
                base_val = float(np.mean(ev))
            else:
                base_val = float(ev)

        # Top-N features por valor absoluto
        top_idx = np.argsort(np.abs(sv))[::-1][:top_n]
        top_features = []
        for i in top_idx:
            fname = feature_names[i] if i < len(feature_names) else f"feature_{i}"
            val   = float(sv[i])
            top_features.append({
                "feature":    fname,
                "shap_value": round(val, 5),
                "direction":  "fraud" if val > 0 else "legit",
            })

        return {
            "predicted_class": predicted_class,
            "top_features":    top_features,
            "base_value":      round(base_val, 5),
            "shap_available":  True,
        }

    except Exception as exc:
        logger.warning(f"SHAP explain_prediction falló: {exc}")
        return {"shap_available": False, "top_features": [], "error": str(exc)}


# ---------------------------------------------------------------------------
# Plot resumen global (beeswarm)
# ---------------------------------------------------------------------------

def plot_shap_summary(
    model: Any,
    X_sample: Any,
    vectorizer: Any,
    use_manual_features: bool,
    model_name: str,
    save_dir: Optional[Path] = None,
    max_display: int = 20,
) -> Optional[Path]:
    """
    Genera y guarda un beeswarm plot de SHAP para el conjunto de evaluación.
    Retorna la ruta del PNG generado, o None si SHAP no está disponible.
    """
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("pip install shap matplotlib para generar plots SHAP.")
        return None

    from src.config import FIGURES_DIR
    if save_dir is None:
        save_dir = FIGURES_DIR

    feature_names = _get_feature_names(vectorizer, use_manual_features)

    # Limitar muestra para velocidad
    X_dense = _to_dense(X_sample)
    if X_dense.shape[0] > 300:
        idx = np.random.default_rng(42).integers(0, X_dense.shape[0], 300)
        X_dense = X_dense[idx]

    try:
        mname = _model_type(model)

        def _extract_2d(raw) -> np.ndarray:
            """Normaliza cualquier formato de shap_values a array 2D (n, features)."""
            if hasattr(raw, "values"):
                v = raw.values
                return v[:, :, -1] if v.ndim == 3 else v
            if isinstance(raw, list):
                return raw[-1]
            return raw[:, :, -1] if raw.ndim == 3 else raw

        if any(t in mname for t in _TREE_MODELS):
            explainer = shap.TreeExplainer(model)
            shap_vals = _extract_2d(explainer.shap_values(X_dense))
        else:
            background = np.zeros((1, X_dense.shape[1]))
            explainer  = shap.LinearExplainer(model, background)
            shap_vals  = _extract_2d(explainer.shap_values(X_dense))

        out_path = save_dir / f"shap_summary_{model_name}.png"
        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            shap_vals,
            X_dense,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP summary plot guardado en {out_path}")
        return out_path

    except Exception as exc:
        logger.warning(f"plot_shap_summary falló: {exc}")
        return None
