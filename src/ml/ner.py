"""
Extracción de entidades nombradas (NER) para mensajes de fraude.

Implementación ligera sin dependencias externas — solo regex.
Extrae teléfonos, URLs, cantidades de dinero y números de cuenta/referencia.
Estas entidades son features con alto valor discriminativo para fraude.
"""

import re
from typing import Any

# ---------------------------------------------------------------------------
# Patrones compilados
# ---------------------------------------------------------------------------

_PHONE_RE = re.compile(
    r"(?<!\d)"
    r"(?:\+?52[\s\-]?)?(?:\d[\s\-]?){7,14}"
    r"(?!\d)",
    re.IGNORECASE,
)

_URL_RE = re.compile(
    r"(?:https?://|www\.)\S+|"
    r"(?:bit\.ly|tinyurl\.com|goo\.gl|t\.co|short\.link)/\S+",
    re.IGNORECASE,
)

_AMOUNT_RE = re.compile(
    r"(?:\$|€|£|MXN|USD|mxn|usd)\s?\d[\d,\.]*|"
    r"\d[\d,\.]*\s?(?:pesos?|dólares?|dollars?|euros?)\b",
    re.IGNORECASE,
)

_ACCOUNT_RE = re.compile(
    r"\b\d{10,18}\b",
)

# Bancos y entidades financieras mencionadas en mensajes de LATAM
_BANK_RE = re.compile(
    r"\b(?:BBVA|Santander|HSBC|Citibanamex|Banamex|Banorte|Scotiabank|"
    r"Inbursa|Azteca|Walmart|Oxxo|Coppel|SAT|IMSS|CFE|Telmex|"
    r"PayPal|MercadoPago|Clip|Nu|Hey\s?Banco|"
    r"Chase|Wells\s?Fargo|Bank\s?of\s?America|Citi|Barclays)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def extract_entities(text: str) -> dict[str, Any]:
    """
    Extrae entidades nombradas relevantes para fraude de un mensaje.

    Returns:
        {
            "phones":        list[str],
            "urls":          list[str],
            "amounts":       list[str],
            "accounts":      list[str],
            "banks":         list[str],
            "entity_count":  int,
            "has_phone":     float,
            "has_url":       float,
            "has_amount":    float,
            "has_bank":      float,
            "has_account":   float,
        }
    """
    phones   = _PHONE_RE.findall(text)
    urls     = _URL_RE.findall(text)
    amounts  = _AMOUNT_RE.findall(text)
    accounts = _ACCOUNT_RE.findall(text)
    banks    = _BANK_RE.findall(text)

    # Filtrar falsos positivos en cuentas (excluir fechas, años cortos)
    accounts = [a for a in accounts if not _is_date_like(a)]

    entity_count = len(phones) + len(urls) + len(amounts) + len(accounts) + len(banks)

    return {
        "phones":       [p.strip() for p in phones],
        "urls":         urls,
        "amounts":      amounts,
        "accounts":     accounts,
        "banks":        banks,
        "entity_count": entity_count,
        "has_phone":    float(bool(phones)),
        "has_url":      float(bool(urls)),
        "has_amount":   float(bool(amounts)),
        "has_bank":     float(bool(banks)),
        "has_account":  float(bool(accounts)),
    }


def _is_date_like(num_str: str) -> bool:
    """Descarta secuencias que son probablemente años o fechas cortas."""
    n = int(num_str.strip())
    return 1900 <= n <= 2100 or len(num_str.strip()) <= 4
