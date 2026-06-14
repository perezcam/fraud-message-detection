"""
Módulo de preprocesamiento textual.
Limpia y normaliza mensajes preservando señales útiles para detección de fraude.
"""

import logging
import re
from typing import Optional

import pandas as pd

from src.config import TEXT_COLUMN

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Patrones de normalización
# ---------------------------------------------------------------------------
_URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+|bit\.ly/\S+|tinyurl\.com/\S+|t\.co/\S+|\S+\.(com|net|org|mx|co|io)/\S*",
    re.IGNORECASE,
)
_EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
)
_PHONE_PATTERN = re.compile(
    r"(\+?\d{1,3}[\s\-\.]?)?(\(?\d{2,4}\)?[\s\-\.]?)?\d{3,4}[\s\-\.]?\d{3,4}(?:[\s\-\.]?\d{0,4})?"
)
_MONEY_PATTERN = re.compile(
    r"(\$|€|£|¥|USD|EUR|MXN|COP|ARS|BRL)\s*[\d,\.]+|"
    r"[\d,\.]+\s*(\$|€|£|¥|USD|EUR|MXN)|"
    r"\b\d[\d,\.]*\s*(pesos?|dólares?|dollars?|euros?)\b",
    re.IGNORECASE,
)
_DUPLICATE_SPACES = re.compile(r"\s+")
# Elimina chars que no aportan información; preserva tokens especiales, puntuación relevante
_IRRELEVANT_CHARS = re.compile(r"[^\wáéíóúüñÁÉÍÓÚÜÑ\s!?.,<>@#$%&*+\-/]", re.UNICODE)


def normalize_urls(text: str) -> str:
    """Reemplaza URLs por el token <URL>."""
    return _URL_PATTERN.sub("<URL>", text)


def normalize_emails(text: str) -> str:
    """Reemplaza correos electrónicos por el token <EMAIL>."""
    return _EMAIL_PATTERN.sub("<EMAIL>", text)


def normalize_phones(text: str) -> str:
    """Reemplaza números de teléfono (≥7 dígitos) por el token <PHONE>."""

    def _replace(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group())
        return "<PHONE>" if len(digits) >= 7 else m.group()

    return _PHONE_PATTERN.sub(_replace, text)


def normalize_money(text: str) -> str:
    """Reemplaza cantidades monetarias por el token <MONEY>."""
    return _MONEY_PATTERN.sub("<MONEY>", text)


def remove_irrelevant_chars(text: str) -> str:
    """Elimina caracteres sin valor informativo sin destruir tokens especiales."""
    return _IRRELEVANT_CHARS.sub(" ", text)


def normalize_spaces(text: str) -> str:
    """Colapsa espacios múltiples en uno solo."""
    return _DUPLICATE_SPACES.sub(" ", text).strip()


def to_lowercase(text: str) -> str:
    """Convierte el texto a minúsculas."""
    return text.lower()


def tokenize(text: str) -> list[str]:
    """Tokenización básica por espacios."""
    return text.split()


def clean_text(text: Optional[str]) -> str:
    """
    Pipeline completo de limpieza para un texto individual.

    Orden: URLs → emails → teléfonos → dinero → minúsculas
           → chars irrelevantes → espacios.

    Preserva: <URL>, <EMAIL>, <PHONE>, <MONEY>, !, ?, signos de puntuación.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = normalize_urls(text)
    text = normalize_emails(text)
    text = normalize_phones(text)
    text = normalize_money(text)
    text = to_lowercase(text)
    text = remove_irrelevant_chars(text)
    text = normalize_spaces(text)
    return text


def preprocess_dataframe(
    df: pd.DataFrame, text_col: str = TEXT_COLUMN
) -> pd.DataFrame:
    """Aplica clean_text a todos los mensajes de un DataFrame."""
    df = df.copy()
    df[text_col] = df[text_col].astype(str).apply(clean_text)
    original_len = len(df)
    df = df[df[text_col].str.strip() != ""]
    removed = original_len - len(df)
    if removed > 0:
        logger.warning(f"Se eliminaron {removed} mensajes vacíos tras el preprocesamiento.")
    logger.info(f"Preprocesamiento completado: {len(df)} mensajes.")
    return df
