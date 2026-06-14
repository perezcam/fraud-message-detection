"""
Extracción de features para ventanas de conversación.

En lugar de contar palabras clave directamente en los scoring functions,
se calcula un vector de features estadísticas y lingüísticas sobre la
secuencia completa. Esto permite que los modelos ML aprendan a partir de
datos, no de reglas escritas a mano.

Features:
  - Trayectoria del riesgo: regresión lineal, varianza, aceleración
  - Distribución temporal: patrones en primera vs segunda mitad
  - Predicciones ML por mensaje: ratio de fraude/legítimo
  - Densidad de señales: pendiente y total de señales individuales
  - Features de vocabulario (agregadas): totales y ratios temporales
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import stats  # regresión lineal

if TYPE_CHECKING:
    from src.conversation.models import Message

from src.config import CREDENTIAL_WORDS, MONEY_WORDS, PRIZE_WORDS, URGENCY_WORDS
from src.rules.risk import PRESSURE_WORDS, TRANSFER_WORDS

# Vocabulario adicional (igual que en patterns.py)
_IMPERSONATION_WORDS = [
    "banco", "bank", "bbva", "santander", "banamex", "hsbc", "citibanamex",
    "scotiabank", "banorte", "imss", "sat", "hacienda", "gobierno",
    "government", "policía", "policia", "police", "fedex", "dhl", "amazon",
    "microsoft", "apple", "paypal", "soporte técnico", "technical support",
    "representante", "ejecutivo", "official", "oficial",
]


def _count_hits(text: str, words: list[str]) -> int:
    t = text.lower()
    return sum(1 for w in words if w in t)


# ---------------------------------------------------------------------------
# Nombres de features (orden fijo para el vector ML)
# ---------------------------------------------------------------------------
FEATURE_NAMES: list[str] = [
    # Trayectoria del riesgo (regresión sobre risk_score de cada mensaje)
    "risk_slope",           # pendiente: positiva = escalada de riesgo
    "risk_delta",           # last_risk - first_risk
    "risk_max",             # máximo riesgo en la ventana
    "risk_mean",            # media de riesgo
    "risk_variance",        # varianza: secuencias erráticas vs uniformes
    "risk_acceleration",    # media segunda mitad - media primera mitad
    "late_risk_spike",      # 1 si último mensaje > media + 1.5·σ

    # Distribución temporal del riesgo
    "early_risk_mean",      # media riesgo primera mitad
    "late_risk_mean",       # media riesgo segunda mitad

    # Predicciones ML por mensaje
    "ml_fraud_ratio",       # fracción de mensajes clasificados como fraude
    "ml_fraud_late_ratio",  # fracción de la 2ª mitad clasificada como fraude
    "ml_legit_early_ratio", # fracción de la 1ª mitad clasificada como legítima

    # Densidad de señales
    "signal_total",         # total de señales individuales en la ventana
    "signal_slope",         # pendiente de señales por mensaje
    "signal_last",          # señales en el último mensaje

    # Vocabulario — totales
    "urgency_count",
    "credential_count",
    "impersonation_count",
    "prize_count",
    "pressure_count",
    "transfer_count",

    # Vocabulario — distribución temporal (ratio late/total, 0 si total=0)
    "urgency_late_ratio",
    "credential_late_ratio",
    "prize_early_ratio",    # premios se anuncian al inicio del scam

    # Estructural
    "n_messages",
]


class WindowFeatureExtractor:
    """
    Extrae un vector de features de una ventana de mensajes.

    Los mensajes deben estar enriquecidos (individual_risk no None) para
    que los features de riesgo y predicción ML sean útiles; si no están
    enriquecidos se usan valores 0.
    """

    def extract(self, messages: list["Message"]) -> dict:
        """Devuelve un dict con todos los features nombrados en FEATURE_NAMES."""
        n = len(messages)

        # --- Riesgo por mensaje ---
        risk = np.array(
            [
                m.individual_risk.get("risk_score", 0) if m.individual_risk else 0
                for m in messages
            ],
            dtype=float,
        )
        half = max(1, n // 2)
        early_risk = risk[:half]
        late_risk  = risk[half:]

        # Regresión lineal sobre la trayectoria del riesgo
        x = np.arange(n, dtype=float)
        if n >= 2:
            slope, _, _, _, _ = stats.linregress(x, risk)
        else:
            slope = 0.0

        sigma = float(risk.std()) if n > 1 else 0.0
        late_spike = (
            1.0 if n > 2 and risk[-1] > risk.mean() + 1.5 * sigma else 0.0
        )

        # --- Predicciones ML ---
        ml_labels = [
            (m.individual_risk.get("ml_label") if m.individual_risk else None)
            for m in messages
        ]
        ml_late  = ml_labels[half:]
        ml_early = ml_labels[:half]

        n_fraud      = sum(1 for l in ml_labels if l == "fraudulent")
        n_fraud_late = sum(1 for l in ml_late   if l == "fraudulent")
        n_legit_early = sum(1 for l in ml_early if l == "legitimate")

        # --- Señales individuales ---
        sig_counts = np.array(
            [
                len(m.individual_risk.get("signals", [])) if m.individual_risk else 0
                for m in messages
            ],
            dtype=float,
        )
        sig_slope = float(stats.linregress(x, sig_counts).slope) if n >= 2 else 0.0

        # --- Vocabulario ---
        texts = [m.text for m in messages]
        early_text = " ".join(texts[:half])
        late_text  = " ".join(texts[half:])

        urg_total  = sum(_count_hits(t, URGENCY_WORDS)      for t in texts)
        urg_late   = _count_hits(late_text, URGENCY_WORDS)

        cred_total = sum(_count_hits(t, CREDENTIAL_WORDS)   for t in texts)
        cred_late  = _count_hits(late_text, CREDENTIAL_WORDS)

        imp_total  = sum(_count_hits(t, _IMPERSONATION_WORDS) for t in texts)

        prize_total = sum(_count_hits(t, PRIZE_WORDS)       for t in texts)
        prize_early = _count_hits(early_text, PRIZE_WORDS)

        press_total = sum(_count_hits(t, PRESSURE_WORDS)    for t in texts)
        trans_total = sum(_count_hits(t, TRANSFER_WORDS)    for t in texts)

        return {
            "risk_slope":           float(slope),
            "risk_delta":           float(risk[-1] - risk[0]),
            "risk_max":             float(risk.max()),
            "risk_mean":            float(risk.mean()),
            "risk_variance":        float(risk.var()),
            "risk_acceleration":    float(late_risk.mean() - early_risk.mean()),
            "late_risk_spike":      late_spike,
            "early_risk_mean":      float(early_risk.mean()),
            "late_risk_mean":       float(late_risk.mean()),
            "ml_fraud_ratio":       float(n_fraud / n),
            "ml_fraud_late_ratio":  float(n_fraud_late / max(len(ml_late), 1)),
            "ml_legit_early_ratio": float(n_legit_early / max(len(ml_early), 1)),
            "signal_total":         float(sig_counts.sum()),
            "signal_slope":         sig_slope,
            "signal_last":          float(sig_counts[-1]),
            "urgency_count":        float(urg_total),
            "urgency_late_ratio":   float(urg_late / max(urg_total, 1)),
            "credential_count":     float(cred_total),
            "credential_late_ratio": float(cred_late / max(cred_total, 1)),
            "impersonation_count":  float(imp_total),
            "prize_count":          float(prize_total),
            "prize_early_ratio":    float(prize_early / max(prize_total, 1)),
            "pressure_count":       float(press_total),
            "transfer_count":       float(trans_total),
            "n_messages":           float(n),
        }

    def to_vector(self, features: dict) -> np.ndarray:
        """Convierte el dict de features a vector numpy ordenado."""
        return np.array(
            [features.get(k, 0.0) for k in FEATURE_NAMES], dtype=np.float32
        )
