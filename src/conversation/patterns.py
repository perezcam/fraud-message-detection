"""
Patrones conductuales de fraude para análisis de conversaciones.

DISEÑO IMPORTANTE:
  Los score_fn ya NO leen texto ni cuentan palabras directamente.
  Reciben un dict de features pre-computados por WindowFeatureExtractor.
  Esto significa que la detección de patrones está basada en:
    - Estadísticas de la trayectoria del riesgo (regresión lineal)
    - Distribución temporal de predicciones ML por mensaje
    - Ratios de vocabulario normalizados (no conteos crudos)
  Los patrones son por tanto "interpretadores semánticos" sobre
  features estadísticos, no detectores heurísticos de palabras clave.

  El ConversationAnalyzer combina estas puntuaciones con la
  predicción del ConversationWindowClassifier (RandomForest).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


# ---------------------------------------------------------------------------
# Funciones de scoring basadas en features estadísticos
# ---------------------------------------------------------------------------

def score_urgency_escalation(features: dict) -> float:
    """
    Detecta escalada de urgencia usando la trayectoria del riesgo y la
    distribución temporal del vocabulario de urgencia.
    """
    slope     = features.get("risk_slope", 0.0)         # pts/mensaje
    accel     = features.get("risk_acceleration", 0.0)  # delta avg segunda vs primera mitad
    urg_count = features.get("urgency_count", 0.0)
    urg_late  = features.get("urgency_late_ratio", 0.0)  # urgencia concentrada al final

    slope_s = min(max(slope / 8.0, 0.0), 1.0)   # pendiente de 8 pts/msg = score máximo
    accel_s = min(max(accel / 30.0, 0.0), 1.0)  # salto de 30 pts en segunda mitad
    urg_s   = min(urg_count / 3.0, 1.0) * (0.5 + 0.5 * urg_late)  # más peso si urgencia es tardía

    return min(0.40 * slope_s + 0.40 * accel_s + 0.20 * urg_s, 0.92)


def score_credential_harvesting(features: dict) -> float:
    """
    Detecta solicitud escalonada de credenciales usando la concentración
    de peticiones de datos en la segunda mitad de la conversación.
    """
    cred_count = features.get("credential_count", 0.0)
    cred_late  = features.get("credential_late_ratio", 0.0)  # peticiones concentradas al final
    imp_count  = features.get("impersonation_count", 0.0)
    fraud_r    = features.get("ml_fraud_ratio", 0.0)  # predicción ML del modelo entrenado
    urg_count  = features.get("urgency_count", 0.0)

    cred_s = min(cred_count / 2.0, 1.0) * (0.5 + 0.5 * cred_late)  # más peso si es tardío
    imp_s  = min(imp_count / 2.0, 1.0)

    s = 0.40 * cred_s + 0.25 * imp_s + 0.20 * fraud_r + 0.15 * min(urg_count / 3.0, 1.0)
    return min(s, 0.92)


def score_social_engineering(features: dict) -> float:
    """
    Detecta inicio inocente + ataque final usando la distribución de
    predicciones ML entre la primera y segunda mitad de la conversación.
    """
    legit_early = features.get("ml_legit_early_ratio", 0.0)
    fraud_late  = features.get("ml_fraud_late_ratio", 0.0)
    accel       = features.get("risk_acceleration", 0.0)
    late_spike  = features.get("late_risk_spike", 0.0)
    n           = features.get("n_messages", 0.0)

    if n < 4:
        return 0.0

    # El patrón es: inicio legítimo + final fraudulento
    contrast_s = legit_early * fraud_late           # máximo cuando ambos son altos
    accel_s    = min(max(accel / 35.0, 0.0), 1.0)

    return min(0.50 * contrast_s + 0.30 * accel_s + 0.20 * late_spike, 0.92)


def score_impersonation(features: dict) -> float:
    """
    Detecta suplantación de identidad usando hits de entidades oficiales
    combinados con vocabulario de urgencia o credenciales.
    """
    imp   = features.get("impersonation_count", 0.0)
    urg   = features.get("urgency_count", 0.0)
    cred  = features.get("credential_count", 0.0)
    fraud = features.get("ml_fraud_ratio", 0.0)

    imp_s  = min(imp / 2.0, 1.0)
    combo  = min((urg + cred) / 3.0, 1.0)  # impersonación sola = bajo riesgo; + urgencia/cred = alto

    s = 0.45 * imp_s + 0.30 * combo + 0.25 * fraud
    return min(s, 0.92)


def score_prize_scam_sequence(features: dict) -> float:
    """
    Detecta secuencia premio→solicitud usando la distribución temporal del
    vocabulario de premios (anuncio al inicio) vs transferencia/credenciales (al final).
    """
    prize_count = features.get("prize_count", 0.0)
    prize_early = features.get("prize_early_ratio", 1.0)  # default alto si no hay premios
    cred_late   = features.get("credential_late_ratio", 0.0)
    trans       = features.get("transfer_count", 0.0)
    fraud_late  = features.get("ml_fraud_late_ratio", 0.0)

    if prize_count < 1:
        return 0.0

    prize_s = min(prize_count / 2.0, 1.0) * prize_early  # premios en primera mitad
    # La solicitud de datos/dinero debe estar en la segunda mitad
    request_s = min(
        max(cred_late * min(features.get("credential_count", 0) / 1.0, 1.0),
            min(trans / 2.0, 1.0),
            fraud_late * 0.6),
        1.0,
    )

    return min(0.50 * prize_s + 0.50 * request_s, 0.92)


def score_financial_coercion(features: dict) -> float:
    """
    Detecta presión financiera usando vocabulario de amenaza/presión combinado
    con solicitudes de transferencia o pago.
    """
    pressure = features.get("pressure_count", 0.0)
    transfer = features.get("transfer_count", 0.0)
    urg      = features.get("urgency_count", 0.0)
    fraud    = features.get("ml_fraud_ratio", 0.0)

    press_s  = min(pressure / 2.0, 1.0)
    trans_s  = min(transfer / 2.0, 1.0)
    urg_s    = min(urg / 3.0, 1.0)

    return min(0.35 * press_s + 0.35 * trans_s + 0.15 * urg_s + 0.15 * fraud, 0.92)


def score_trust_building_attack(features: dict) -> float:
    """
    Detecta conversación amigable que culmina en ataque usando la asimetría
    entre la primera mitad legítima y el pico de riesgo al final.
    """
    legit_early = features.get("ml_legit_early_ratio", 0.0)
    late_spike  = features.get("late_risk_spike", 0.0)
    risk_delta  = features.get("risk_delta", 0.0)
    n           = features.get("n_messages", 0.0)

    if n < 4:
        return 0.0

    delta_s     = min(max(risk_delta / 60.0, 0.0), 1.0)  # delta de 60 pts = score máximo
    contrast_s  = legit_early * (0.5 * late_spike + 0.5 * delta_s)

    return min(0.60 * contrast_s + 0.40 * delta_s, 0.92)


# ---------------------------------------------------------------------------
# Registro de patrones
# ---------------------------------------------------------------------------

@dataclass
class Pattern:
    name: str
    description: str
    min_window_size: int
    score_fn: Callable[[dict], float]   # recibe features dict, no lista de mensajes


PATTERNS: list[Pattern] = [
    Pattern(
        name="urgency_escalation",
        description="La urgencia aumenta progresivamente a lo largo de los mensajes",
        min_window_size=3,
        score_fn=score_urgency_escalation,
    ),
    Pattern(
        name="credential_harvesting",
        description="Solicitud escalonada de credenciales, códigos o datos personales",
        min_window_size=2,
        score_fn=score_credential_harvesting,
    ),
    Pattern(
        name="social_engineering",
        description="Mensajes iniciales inofensivos que derivan en solicitud fraudulenta",
        min_window_size=4,
        score_fn=score_social_engineering,
    ),
    Pattern(
        name="impersonation",
        description="Suplantación de entidad oficial (banco, gobierno, empresa) + acción requerida",
        min_window_size=2,
        score_fn=score_impersonation,
    ),
    Pattern(
        name="prize_scam_sequence",
        description="Anuncio de premio seguido de solicitud de datos o pago para reclamarlo",
        min_window_size=2,
        score_fn=score_prize_scam_sequence,
    ),
    Pattern(
        name="financial_coercion",
        description="Presión o amenaza combinada con solicitud de transferencia o pago",
        min_window_size=2,
        score_fn=score_financial_coercion,
    ),
    Pattern(
        name="trust_building_attack",
        description="Conversación aparentemente inofensiva que culmina en solicitud fraudulenta",
        min_window_size=4,
        score_fn=score_trust_building_attack,
    ),
]
