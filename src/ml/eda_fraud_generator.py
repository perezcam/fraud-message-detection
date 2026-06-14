"""
Generador de mensajes de fraude sintéticos mediante EDA (Estimation of Distribution Algorithm).

Fuente: Metaheurísticas (INFOS/4- Metaheuristicas.pdf) — Algoritmos de Estimación de Distribuciones.

En lugar de copiar mensajes del dataset, el EDA aprende la distribución probabilística
de las características de los mensajes de fraude:
  P(palabra_k | fraude)  → probabilidad de que la palabra k aparezca en un fraude
  P(feature_j | fraude)  → distribución de cada feature manual dado que es fraude

Luego muestrea de esa distribución para generar nuevos vectores que "suenan a fraude"
y los decodifica a texto mediante templates en español.

Ventajas:
  - Sin API externa (no requiere Mistral)
  - Resuelve el desbalance 747 fraude / 4,825 legítimo
  - Los mensajes generados siguen la distribución estadística real, no plantillas fijas

Uso:
    gen = EDAFraudGenerator()
    info = gen.fit(df, vectorizer)
    texts = gen.generate_texts(n=500)
    df_syn = gen.generate_dataframe(n=500)
    gen.save()
"""

import logging
import random
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.config import (
    LABEL_COLUMN,
    MODELS_DIR,
    TEXT_COLUMN,
    URGENCY_WORDS,
    CREDENTIAL_WORDS,
    MONEY_WORDS,
)
from src.data.preprocessing import clean_text

logger = logging.getLogger(__name__)

EDA_GENERATOR_FILE = "eda_generator.joblib"

# ---------------------------------------------------------------------------
# Templates en español para decodificación de texto
# Los marcadores {X} son reemplazados con palabras de alta P(word|fraud)
# ---------------------------------------------------------------------------

_TEMPLATES = [
    "ALERTA {banco}: su cuenta ha sido {accion}. {urgencia} en {url} o llame al {phone}.",
    "Estimado cliente de {banco}, su {servicio} será {accion} si no {urgencia} hoy.",
    "Su {servicio} de {banco} está {accion}. Proporcione su {credencial} en {url}.",
    "{banco} le informa: detectamos actividad inusual. Verifique en {url} con su {credencial}.",
    "URGENTE: su {servicio} vence en 24 horas. Deposite {monto} en {url} para conservarlo.",
    "Felicidades, fue seleccionado para recibir {monto}. Confirme sus datos en {url}.",
    "Su {credencial} fue comprometida. Cambie su contraseña inmediatamente en {url}.",
    "Paquete retenido: para liberar su envío deposite {monto} en {url} o al {phone}.",
    "El SAT detectó irregularidades en su RFC. Regularícese en {url} con su {credencial}.",
    "OFERTA EXCLUSIVA: {monto} de reembolso disponible. Reclamelo en {url} hoy.",
    "Su cuenta {banco} fue bloqueada por seguridad. Verifique con su {credencial} en {url}.",
    "Notificación: intento de acceso no autorizado. Confirme identidad en {url} urgente.",
]

_BANCOS    = ["BBVA", "Santander", "HSBC", "Banamex", "Citibanamex", "Banorte",
              "Scotiabank", "el banco", "su institución bancaria", "SAT"]
_SERVICIOS = ["cuenta", "tarjeta", "crédito", "servicio", "membresía", "acceso", "cuenta bancaria"]
_ACCIONES  = ["bloqueada", "suspendida", "comprometida", "limitada", "vencida",
               "cancelada", "restringida", "en revisión"]
_URLS      = ["bit.ly/verificar", "bbva-seguro.com", "sat-tramites.mx",
               "verificacion-cuenta.com", "banco-seguro.net", "tramite-urgente.com"]
_PHONES    = ["800 123 4567", "55 1234 5678", "+52 55 9876 5432", "01 800 000 1234"]
_MONTOS    = ["$5,000 MXN", "$500 pesos", "50% de descuento", "un depósito de $1,500"]
_CREDS     = ["NIP", "contraseña", "código OTP", "número de tarjeta", "credenciales", "PIN"]


class EDAFraudGenerator:
    """
    Generador de mensajes de fraude sintéticos basado en EDA.

    Aprende la distribución de palabras y features en los mensajes de fraude
    del dataset y genera nuevos mensajes mediante muestreo de esa distribución.

    Uso:
        gen = EDAFraudGenerator()
        gen.fit(df, vectorizer)
        gen.save()

        # Generar mensajes sintéticos:
        df_syn = gen.generate_dataframe(n=500)
        # Concatenar con dataset original para rebalanceo
    """

    def __init__(self, random_state: int = 42) -> None:
        self._rng          = random.Random(random_state)
        self._np_rng       = np.random.default_rng(random_state)
        self._top_words:   list[str]          = []
        self._word_probs:  dict[str, float]   = {}
        self._fitted       = False

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, vectorizer) -> dict:
        """
        Estima la distribución de features sobre los mensajes de fraude.

        Args:
            df:         DataFrame con TEXT_COLUMN y LABEL_COLUMN.
            vectorizer: TF-IDF vectorizador ya ajustado.

        Returns:
            {"n_fraud": int, "n_features": int, "top_fraud_words": list[str]}
        """
        df_fraud = df[df[LABEL_COLUMN] == "fraudulent"].copy()
        if len(df_fraud) == 0:
            raise ValueError("El dataset no tiene mensajes de fraude.")

        texts = df_fraud[TEXT_COLUMN].astype(str).apply(clean_text).tolist()

        # Distribución de palabras P(word | fraude)
        X_fraud = vectorizer.transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        # Para cada palabra: P(aparece en un mensaje de fraude)
        word_freq = np.asarray(X_fraud.sum(axis=0)).flatten()
        total     = float(len(texts))

        self._word_probs = {
            feature_names[i]: float(word_freq[i] / total)
            for i in range(len(feature_names))
            if word_freq[i] > 0
        }

        # Top palabras por probabilidad (excluir stopwords muy comunes)
        _STOPWORDS = {"de", "la", "el", "en", "a", "y", "que", "se", "su", "un",
                      "una", "los", "las", "es", "por", "con", "para", "al", "del"}
        sorted_words = sorted(self._word_probs.items(), key=lambda x: x[1], reverse=True)
        self._top_words = [w for w, _ in sorted_words
                           if len(w) > 3 and w not in _STOPWORDS][:200]

        self._fitted = True
        n_feats = len(feature_names)

        logger.info(
            f"EDAFraudGenerator ajustado: {len(df_fraud)} fraudes, "
            f"{n_feats} features, {len(self._top_words)} palabras top"
        )
        return {
            "n_fraud":        len(df_fraud),
            "n_features":     n_feats,
            "top_fraud_words": self._top_words[:20],
        }

    # ------------------------------------------------------------------
    # Generación de texto
    # ------------------------------------------------------------------

    def generate_texts(self, n: int = 100) -> list[str]:
        """
        Genera n mensajes de fraude sintéticos usando templates y distribución aprendida.

        Returns:
            list[str] de textos sintéticos etiquetados implícitamente como fraude.
        """
        if not self._fitted:
            raise RuntimeError("EDAFraudGenerator no ajustado — llame a fit() primero.")

        results = []
        for _ in range(n):
            template = self._rng.choice(_TEMPLATES)
            text = self._fill_template(template)
            results.append(text)
        return results

    def generate_dataframe(self, n: int = 100) -> pd.DataFrame:
        """Genera n mensajes y los empaqueta con label="fraudulent"."""
        texts = self.generate_texts(n)
        return pd.DataFrame({TEXT_COLUMN: texts, LABEL_COLUMN: "fraudulent"})

    # ------------------------------------------------------------------
    # Llenado de templates
    # ------------------------------------------------------------------

    def _fill_template(self, template: str) -> str:
        """Reemplaza marcadores del template con valores del vocabulario aprendido."""
        # Selecciona palabras de alta probabilidad del vocabulario de fraude
        fraud_vocab = self._top_words[:50] if len(self._top_words) >= 50 else self._top_words

        replacements = {
            "{banco}":      self._rng.choice(_BANCOS),
            "{servicio}":   self._rng.choice(_SERVICIOS),
            "{accion}":     self._rng.choice(_ACCIONES),
            "{url}":        self._rng.choice(_URLS),
            "{phone}":      self._rng.choice(_PHONES),
            "{monto}":      self._rng.choice(_MONTOS),
            "{credencial}": self._rng.choice(_CREDS),
            "{urgencia}":   self._rng.choice(URGENCY_WORDS[:10]),
        }

        text = template
        for marker, value in replacements.items():
            text = text.replace(marker, value)

        # Enriquece con 1-3 palabras de alta probabilidad del vocabulario de fraude
        if fraud_vocab:
            extra_count = self._rng.randint(1, 3)
            extras = self._rng.sample(fraud_vocab, min(extra_count, len(fraud_vocab)))
            text = text + " " + " ".join(extras)

        return text.strip()

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> Path:
        if not self._fitted:
            raise RuntimeError("EDAFraudGenerator no ajustado.")
        out = Path(path) if path else MODELS_DIR / EDA_GENERATOR_FILE
        joblib.dump({
            "top_words":  self._top_words,
            "word_probs": self._word_probs,
        }, out)
        logger.info(f"EDAFraudGenerator guardado en {out}")
        return out

    def load(self, path: Optional[Path] = None) -> None:
        p = Path(path) if path else MODELS_DIR / EDA_GENERATOR_FILE
        if not p.exists():
            raise FileNotFoundError(f"EDAFraudGenerator no encontrado: {p}")
        data = joblib.load(p)
        self._top_words  = data["top_words"]
        self._word_probs = data["word_probs"]
        self._fitted     = True
        logger.info(f"EDAFraudGenerator cargado desde {p}")
