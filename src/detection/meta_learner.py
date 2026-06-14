"""
Meta-learner de stacking para el CascadePredictor.

En lugar de combinar las capas con max() (conservador, sin aprendizaje),
un LightGBM aprende los pesos óptimos entre las señales de cada capa.

Meta-features de entrada (5 dimensiones):
  0  ml_proba_fraud     — probabilidad ML de clase fraudulenta (0-1)
  1  rule_score_norm    — rule_score / 100
  2  emb_fraud_sim      — similitud coseno con ejemplos fraudulentos (0 si no disponible)
  3  emb_legit_sim      — similitud coseno con ejemplos legítimos (0 si no disponible)
  4  anomaly_score      — score de Isolation Forest (0 si no disponible)

El meta-learner se entrena sobre el dataset procesado:
  1. Para cada mensaje, se corren las capas ML + reglas (+ anomaly si disponible)
  2. Se construye el vector de 5 meta-features
  3. Se entrena LightGBM (n_estimators=50 para evitar overfitting sobre meta-features)
  4. Se evalúa en un split de prueba interno

En inferencia (cascade.py), si el meta-learner está disponible:
  - Su predict_proba() reemplaza la lógica _aggregate() simple
  - Si NO está disponible, cascade.py sigue con la lógica original

Guardado en models/meta_learner.joblib
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.config import LABEL_COLUMN, MODELS_DIR, RANDOM_STATE, TEXT_COLUMN
from src.data.preprocessing import clean_text

logger = logging.getLogger(__name__)

META_FILE = "meta_learner.joblib"
META_FEATURE_NAMES = [
    "ml_proba_fraud",
    "rule_score_norm",
    "emb_fraud_sim",
    "emb_legit_sim",
    "anomaly_score",
    "transformer_proba",
    "bayes_score",
    "cbr_score",
]


def _binary_label(label: str) -> int:
    return 1 if label == "fraudulent" else 0


def _extract_meta_features(
    cascade,
    message: str,
) -> np.ndarray:
    """
    Extrae el vector de meta-features para un mensaje usando las capas disponibles.
    No invoca LLM (demasiado lento para entrenamiento batch).
    """
    # Capa ML + reglas (siempre disponibles)
    ml = cascade._ml.predict(message)
    ml_conf = ml.get("confidence") or 0.0
    # Convertir confianza a probabilidad de clase fraudulenta
    pred_class = ml.get("predicted_class", "legitimate")
    if pred_class == "fraudulent":
        ml_proba = ml_conf
    else:
        ml_proba = 1.0 - ml_conf if ml_conf is not None else 0.5

    rule_norm = ml.get("risk_score", 0) / 100.0

    # Capa embeddings (opcional)
    emb_fraud_sim = 0.0
    emb_legit_sim = 0.0
    if cascade._emb is not None:
        try:
            emb = cascade._emb.search(message)
            emb_fraud_sim = emb.get("fraud_similarity", 0.0)
            emb_legit_sim = emb.get("legit_similarity", 0.0)
        except Exception:
            pass

    # Capa anomalía (opcional)
    anomaly = 0.0
    if cascade._anomaly is not None:
        try:
            anomaly = cascade._anomaly.score(message)
        except Exception:
            pass

    # Capa transformer (opcional)
    transformer = 0.0
    if cascade._transformer is not None:
        try:
            transformer = cascade._transformer.predict_proba(message)
        except Exception:
            pass

    # Red Bayesiana (opcional)
    bayes = 0.0
    if cascade._bayes is not None:
        try:
            bayes = cascade._bayes.score_message(message)
        except Exception:
            pass

    # CaseBase CBR (opcional)
    cbr = 0.0
    if cascade._case_base is not None:
        try:
            cbr = cascade._case_base.query(message)["cbr_score"]
        except Exception:
            pass

    return np.array(
        [ml_proba, rule_norm, emb_fraud_sim, emb_legit_sim, anomaly, transformer, bayes, cbr],
        dtype=np.float32,
    )


class CascadeMetaLearner:
    """
    Meta-learner LightGBM que combina las señales del CascadePredictor.

    Uso:
        meta = CascadeMetaLearner()
        metrics = meta.fit(df, cascade, sample=2000)
        meta.save()

        # En inferencia (desde cascade.py):
        meta.load()
        score = meta.predict_proba(meta_features_dict)
    """

    def __init__(self) -> None:
        self._model = None
        self._fitted = False
        self._threshold = 0.5

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        cascade,
        sample: int = 2000,
        test_size: float = 0.20,
    ) -> dict:
        """
        Genera meta-features para cada mensaje y entrena el meta-learner.

        Args:
            df:      DataFrame con columnas "message" y "label".
            cascade: Instancia de CascadePredictor ya inicializada.
            sample:  Máximo de mensajes a usar (estratificado por clase).
            test_size: Fracción para evaluación interna.

        Returns:
            dict con accuracy, auc, train_n del meta-learner.
        """
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError("pip install lightgbm") from exc

        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, roc_auc_score

        # Muestreo estratificado
        frames = []
        for label, grp in df.groupby(LABEL_COLUMN):
            n = min(len(grp), sample // 2)
            frames.append(grp.sample(n, random_state=RANDOM_STATE))
        df_sample = pd.concat(frames, ignore_index=True).sample(
            frac=1, random_state=RANDOM_STATE
        )

        logger.info(
            f"Generando meta-features para {len(df_sample)} mensajes "
            f"(embeddings={'disponible' if cascade._emb else 'no'}, "
            f"anomaly={'disponible' if cascade._anomaly else 'no'})..."
        )

        X_meta = []
        y_meta = []
        for _, row in df_sample.iterrows():
            try:
                feats = _extract_meta_features(cascade, str(row[TEXT_COLUMN]))
                X_meta.append(feats)
                y_meta.append(_binary_label(row[LABEL_COLUMN]))
            except Exception as exc:
                logger.debug(f"Error en meta-features: {exc}")

        X_meta = np.array(X_meta, dtype=np.float32)
        y_meta = np.array(y_meta, dtype=int)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_meta, y_meta,
            test_size=test_size,
            random_state=RANDOM_STATE,
            stratify=y_meta,
        )

        fraud_n = y_tr.sum()
        legit_n = len(y_tr) - fraud_n
        scale_pos = legit_n / max(fraud_n, 1)

        self._model = LGBMClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale_pos,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
        self._model.fit(X_tr, y_tr, feature_name=META_FEATURE_NAMES)
        self._fitted = True

        y_pred = self._model.predict(X_te)
        y_prob = self._model.predict_proba(X_te)[:, 1]
        acc = float(accuracy_score(y_te, y_pred))
        auc = float(roc_auc_score(y_te, y_prob))

        logger.info(
            f"Meta-learner entrenado: {len(X_tr)} train / {len(X_te)} test — "
            f"accuracy={acc:.4f}, AUC={auc:.4f}"
        )
        return {"accuracy": acc, "auc": auc, "train_n": len(X_tr)}

    # ------------------------------------------------------------------
    # Inferencia
    # ------------------------------------------------------------------

    def predict_proba_from_dict(self, meta_features: dict) -> float:
        """
        Recibe un dict con las meta-features ya calculadas y retorna
        la probabilidad de que el mensaje sea fraudulento (0-1).
        """
        if not self._fitted:
            raise RuntimeError("Meta-learner no entrenado.")
        x = np.array(
            [[meta_features.get(k, 0.0) for k in META_FEATURE_NAMES]],
            dtype=np.float32,
        )
        return float(self._model.predict_proba(x)[0, 1])

    def predict_proba_from_array(self, feats: np.ndarray) -> float:
        if not self._fitted:
            raise RuntimeError("Meta-learner no entrenado.")
        return float(self._model.predict_proba(feats.reshape(1, -1))[0, 1])

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> Path:
        if not self._fitted:
            raise RuntimeError("Meta-learner no entrenado.")
        out = Path(path) if path else MODELS_DIR / META_FILE
        joblib.dump({"model": self._model, "threshold": self._threshold}, out)
        logger.info(f"Meta-learner guardado en {out}")
        return out

    def load(self, path: Optional[Path] = None) -> None:
        p = Path(path) if path else MODELS_DIR / META_FILE
        if not p.exists():
            raise FileNotFoundError(f"Meta-learner no encontrado: {p}")
        data = joblib.load(p)
        self._model     = data["model"]
        self._threshold = data.get("threshold", 0.5)
        self._fitted    = True
        logger.info(f"Meta-learner cargado desde {p}")
