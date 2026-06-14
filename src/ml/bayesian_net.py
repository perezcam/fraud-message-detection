"""
Red Bayesiana para detección de fraude.

Fuente: Sistemas Expertos (INFOS/7- Sistemas Expertos.pdf), sección Redes Bayesianas.

En lugar de sumar señales con pesos fijos (como el rule_score actual),
una Red Bayesiana modela las DEPENDENCIAS CONDICIONALES entre señales:
  - La urgencia sola es poco sospechosa (ej: "tu cita es urgente")
  - La urgencia + mención de banco + solicitud de credencial = muy sospechoso
  - P(fraude | urgencia=1, banco=1, credencial=1) >> P(fraude | urgencia=1)

Estructura del grafo:
  url_present ─────────────────────────────────────┐
  urgency ──────────────────────────────────────────┤
  credential_request ──────────────────────────────→│ fraud
  bank_mentioned ───────────────────────────────────┤
  amount_present ───────────────────────────────────┤
  negation_present ─────────────────────────────────┘

Librería: pgmpy (pura Python, sin GPU, sin dependencias pesadas)
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.config import (
    BAYES_NET_FILE,
    CREDENTIAL_WORDS,
    LABEL_COLUMN,
    MODELS_DIR,
    MONEY_WORDS,
    NEGATION_WORDS,
    TEXT_COLUMN,
    URGENCY_WORDS,
)
from src.data.preprocessing import clean_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extracción de evidencia binaria para la red
# ---------------------------------------------------------------------------

_BANK_WORDS = [
    "bbva", "santander", "hsbc", "banamex", "citibanamex", "banorte",
    "scotiabank", "inbursa", "azteca", "sat", "imss", "cfe",
    "banco", "bank", "paypal", "mercadopago",
]

_AMOUNT_RE_STR = r"(?:\$|€|£|mxn|usd)\s?\d|\d[\d,\.]*\s?(?:pesos?|dólares?|dollars?)"


def extract_bayes_features(text: str) -> dict[str, int]:
    """Extrae 6 variables binarias que alimentan la red Bayesiana."""
    import re
    tl = text.lower()
    tokens = tl.split()

    has_url        = int(bool(re.search(r"https?://|www\.|bit\.ly|tinyurl", tl)))
    has_urgency    = int(any(w in tl for w in URGENCY_WORDS))
    has_credential = int(any(w in tl for w in CREDENTIAL_WORDS))
    has_bank       = int(any(w in tl for w in _BANK_WORDS))
    has_amount     = int(bool(re.search(_AMOUNT_RE_STR, tl, re.IGNORECASE)))
    has_negation   = int(any(w in tokens for w in NEGATION_WORDS))

    return {
        "url_present":        has_url,
        "urgency":            has_urgency,
        "credential_request": has_credential,
        "bank_mentioned":     has_bank,
        "amount_present":     has_amount,
        "negation_present":   has_negation,
    }


# ---------------------------------------------------------------------------
# Red Bayesiana
# ---------------------------------------------------------------------------

class FraudBayesNet:
    """
    Red Bayesiana naive (Naive Bayes estructurado) para fraude.

    Usa estimación MLE con suavizado de Laplace para las tablas de
    probabilidad condicional (CPTs). No requiere pgmpy para inferencia —
    implementación directa más robusta y sin dependencias externas.

    Uso:
        bn = FraudBayesNet()
        bn.fit(df)
        bn.save()

        # En inferencia:
        bn.load()
        score = bn.predict_proba({"url_present": 1, "urgency": 1, ...})
    """

    FEATURES = [
        "url_present", "urgency", "credential_request",
        "bank_mentioned", "amount_present", "negation_present",
    ]

    def __init__(self) -> None:
        self._cpts: dict = {}       # P(feature=1 | fraud=c) por clase
        self._prior: dict = {}      # P(fraud=c)
        self._fitted = False

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> dict:
        """
        Estima las CPTs por máxima verosimilitud con suavizado de Laplace.

        Returns:
            dict con estadísticas del entrenamiento.
        """
        df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
        df = df[df[LABEL_COLUMN].isin(["fraudulent", "legitimate"])].copy()
        df["_label_bin"] = (df[LABEL_COLUMN] == "fraudulent").astype(int)

        logger.info(f"Entrenando FraudBayesNet con {len(df)} mensajes...")

        # Extraer features binarias
        feat_rows = df[TEXT_COLUMN].astype(str).apply(clean_text).apply(extract_bayes_features)
        X = pd.DataFrame(feat_rows.tolist())
        y = df["_label_bin"].values

        # Prior: P(fraud=1) y P(fraud=0)
        n_fraud = y.sum()
        n_legit = len(y) - n_fraud
        self._prior = {
            1: (n_fraud + 1) / (len(y) + 2),   # Laplace
            0: (n_legit + 1) / (len(y) + 2),
        }

        # CPTs: P(feature_k = 1 | fraud = c)
        self._cpts = {}
        for feat in self.FEATURES:
            self._cpts[feat] = {}
            for c in [0, 1]:
                mask = (y == c)
                n_c      = mask.sum()
                n_feat_c = X.loc[mask, feat].sum()
                # Laplace smoothing
                self._cpts[feat][c] = (n_feat_c + 1) / (n_c + 2)

        self._fitted = True

        # Calcular AUC aproximado en el mismo dataset (solo indicativo)
        scores = [self.predict_proba(extract_bayes_features(str(t)))
                  for t in df[TEXT_COLUMN]]
        from sklearn.metrics import roc_auc_score
        auc = round(float(roc_auc_score(y, scores)), 4)
        logger.info(f"FraudBayesNet entrenado — AUC (train): {auc}")
        return {"train_n": len(df), "auc_train": auc,
                "prior_fraud": round(self._prior[1], 4)}

    # ------------------------------------------------------------------
    # Inferencia
    # ------------------------------------------------------------------

    def predict_proba(self, features: dict) -> float:
        """
        Calcula P(fraud=1 | evidencia) usando el teorema de Bayes naive.

        features: dict con las mismas claves que FEATURES (valores 0 o 1).
        """
        if not self._fitted:
            raise RuntimeError("FraudBayesNet no entrenado.")

        log_fraud = np.log(self._prior[1])
        log_legit = np.log(self._prior[0])

        for feat in self.FEATURES:
            val = int(features.get(feat, 0))
            p_feat_fraud = self._cpts[feat][1]
            p_feat_legit = self._cpts[feat][0]

            if val == 1:
                log_fraud += np.log(p_feat_fraud)
                log_legit += np.log(p_feat_legit)
            else:
                log_fraud += np.log(1 - p_feat_fraud)
                log_legit += np.log(1 - p_feat_legit)

        # Softmax para obtener probabilidad normalizada
        max_log = max(log_fraud, log_legit)
        exp_fraud = np.exp(log_fraud - max_log)
        exp_legit = np.exp(log_legit - max_log)
        return round(float(exp_fraud / (exp_fraud + exp_legit)), 4)

    def score_message(self, message: str) -> float:
        """Atajo: limpia el texto y retorna P(fraud)."""
        clean = clean_text(message) or message
        return self.predict_proba(extract_bayes_features(clean))

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> Path:
        if not self._fitted:
            raise RuntimeError("Red no entrenada.")
        out = Path(path) if path else MODELS_DIR / BAYES_NET_FILE
        joblib.dump({"cpts": self._cpts, "prior": self._prior}, out)
        logger.info(f"FraudBayesNet guardado en {out}")
        return out

    def load(self, path: Optional[Path] = None) -> None:
        p = Path(path) if path else MODELS_DIR / BAYES_NET_FILE
        if not p.exists():
            raise FileNotFoundError(f"FraudBayesNet no encontrado: {p}")
        data = joblib.load(p)
        self._cpts   = data["cpts"]
        self._prior  = data["prior"]
        self._fitted = True
        logger.info(f"FraudBayesNet cargado desde {p}")
