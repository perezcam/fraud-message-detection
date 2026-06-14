"""
Módulo de análisis de riesgo basado en reglas.
Detecta señales sospechosas y calcula un puntaje de riesgo heurístico.
"""

import re
import logging

from src.config import (
    CREDENTIAL_WORDS,
    MONEY_WORDS,
    PRIZE_WORDS,
    RISK_SCORE_HIGH_THRESHOLD,
    RISK_SCORE_MEDIUM_THRESHOLD,
    URGENCY_WORDS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Patrones de detección
# ---------------------------------------------------------------------------
_URL_RE = re.compile(
    r"https?://\S+|www\.\S+|bit\.ly/\S+|t\.co/\S+|\S+\.(com|net|org|mx)/\S*",
    re.IGNORECASE,
)
_PHONE_RE = re.compile(
    r"(\+?\d{1,3}[\s\-\.]?)?(\(?\d{2,4}\)?[\s\-\.]?)?\d{3,4}[\s\-\.]?\d{3,4}"
)
_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
)
_MONEY_RE = re.compile(
    r"(\$|€|£|¥)\s*[\d,\.]+|[\d,\.]+\s*(\$|€|£|¥)|"
    r"\b\d[\d,\.]*\s*(pesos?|dólares?|dollars?|euros?)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Vocabularios adicionales específicos de fraude
# ---------------------------------------------------------------------------
ACCOUNT_BLOCK_WORDS = [
    "bloqueado", "blocked", "suspendido", "suspended", "desactivado",
    "deactivated", "restringido", "restricted", "cerrado", "closed",
    "cancelado", "cancelled", "vencido", "expired", "limite", "limit",
    "inhabilitado", "disabled",
]

TRANSFER_WORDS = [
    "transferencia", "transfer", "transferir", "depositar", "deposit",
    "enviar dinero", "send money", "pagar", "pay", "cobro", "cargo",
    "reembolso", "refund", "reintegro", "abono", "remesa", "remittance",
    "giro", "wire", "pago inmediato",
]

PRESSURE_WORDS = [
    "última oportunidad", "last chance", "no ignore", "no ignores",
    "imprescindible", "obligatorio", "mandatory", "requerido", "required",
    "inmediatamente", "immediately", "de lo contrario", "otherwise",
    "si no actúa", "si no responde", "perderá su cuenta", "will lose",
    "multa", "fine", "sanción", "penalty", "será eliminado", "will be deleted",
    "acción requerida", "action required", "responda ahora", "reply now",
]


def _has_pattern(text: str, pattern: re.Pattern) -> bool:
    return bool(pattern.search(text))


def _count_pattern(text: str, pattern: re.Pattern) -> int:
    return len(pattern.findall(text))


def _has_words(text: str, word_list: list[str]) -> bool:
    text_lower = text.lower()
    return any(word in text_lower for word in word_list)


def analyze_risk(text: str) -> dict:
    """
    Analiza un mensaje y devuelve señales detectadas y puntaje de riesgo.

    Returns:
        {
            "risk_score": int (0-100),
            "risk_level": "low" | "medium" | "high",
            "signals": list[str]
        }
    """
    if not isinstance(text, str) or not text.strip():
        return {"risk_score": 0, "risk_level": "low", "signals": []}

    signals: list[str] = []
    score = 0

    # URLs
    url_count = _count_pattern(text, _URL_RE)
    if url_count > 0:
        signals.append(f"Contiene URL{'s' if url_count > 1 else ''} ({url_count})")
        score += 20

    # Teléfonos (findall con grupos devuelve tuplas; usar finditer para el match completo)
    phone_digits = [re.sub(r"\D", "", m.group()) for m in _PHONE_RE.finditer(text)]
    if any(len(d) >= 7 for d in phone_digits):
        signals.append("Contiene número de teléfono")
        score += 10

    # Correos electrónicos
    if _has_pattern(text, _EMAIL_RE):
        signals.append("Contiene dirección de correo electrónico")
        score += 10

    # Cantidades monetarias
    if _has_pattern(text, _MONEY_RE):
        signals.append("Menciona cantidades monetarias")
        score += 15

    # Urgencia
    if _has_words(text, URGENCY_WORDS):
        signals.append("Usa lenguaje de urgencia")
        score += 20

    # Bloqueo / suspensión de cuenta
    if _has_words(text, ACCOUNT_BLOCK_WORDS):
        signals.append("Amenaza con bloqueo o suspensión de cuenta")
        score += 20

    # Premios / ofertas
    if _has_words(text, PRIZE_WORDS):
        signals.append("Menciona premios, regalos u ofertas")
        score += 15

    # Transferencias / pagos
    if _has_words(text, TRANSFER_WORDS):
        signals.append("Solicita transferencia o pago")
        score += 20

    # Credenciales / códigos / PIN / OTP
    if _has_words(text, CREDENTIAL_WORDS):
        signals.append("Solicita contraseña, código, PIN u OTP")
        score += 25

    # Lenguaje de presión
    if _has_words(text, PRESSURE_WORDS):
        signals.append("Usa lenguaje de presión o amenaza")
        score += 20

    # Palabras monetarias genéricas
    if _has_words(text, MONEY_WORDS):
        if not _has_pattern(text, _MONEY_RE):  # ya contado arriba si hay monto exacto
            signals.append("Menciona términos financieros o bancarios")
            score += 10

    # Signos de exclamación excesivos
    if text.count("!") >= 3:
        signals.append("Uso excesivo de signos de exclamación")
        score += 5

    score = min(score, 100)

    if score >= RISK_SCORE_HIGH_THRESHOLD:
        risk_level = "high"
    elif score >= RISK_SCORE_MEDIUM_THRESHOLD:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {"risk_score": score, "risk_level": risk_level, "signals": signals}
