"""
Generador de ejemplos adversariales para robustecimiento del sistema.

Fuente: Búsqueda Adversarial (INFOS/2 Busqueda Adversarial.pptx).

Inspiración adversarial (Minimax): el "adversario" intenta generar mensajes
de fraude que el detector clasifique como legítimos (evasión). Al entrenar
con estos ejemplos, el modelo se vuelve más robusto.

Perturbaciones implementadas:
  1. DILUTION     — inserta frases de saludo/legítimas al inicio
  2. SYNONYM      — reemplaza palabras de urgencia por sinónimos menos obvios
  3. NEGATION_WRAP — envuelve el mensaje en contexto de negación falso
  4. SPACING      — separa caracteres en palabras clave (ob-ligatorio)
  5. TYPO         — introduce typos controlados en palabras de señal

Los adversariales generados se etiquetan como "fraudulent" y se pueden
añadir al dataset de entrenamiento para aumentar la robustez.
"""

import logging
import random
import re
from typing import Optional

import pandas as pd

from src.config import (
    CREDENTIAL_WORDS,
    LABEL_COLUMN,
    TEXT_COLUMN,
    URGENCY_WORDS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Perturbaciones
# ---------------------------------------------------------------------------

_LEGIT_GREETINGS = [
    "Hola, espero que estés bien. ",
    "Buenos días. ",
    "Estimado cliente, ",
    "Como siempre, con mucho gusto le atendemos. ",
    "Un cordial saludo. ",
]

_LEGIT_CLOSINGS = [
    " Que tenga un buen día.",
    " Saludos cordiales.",
    " Quedamos a sus órdenes.",
    " Gracias por su preferencia.",
]

_URGENCY_SYNONYMS = {
    "urgente":       ["importante", "prioritario", "crítico"],
    "urgent":        ["important", "critical", "essential"],
    "inmediatamente": ["pronto", "en breve", "lo antes posible"],
    "immediately":   ["soon", "promptly", "right away"],
    "ahora":         ["en este momento", "actualmente", "hoy"],
    "now":           ["at this time", "today", "currently"],
    "rápido":        ["breve", "rápidamente", "sin tardanza"],
    "quickly":       ["promptly", "briefly"],
    "suspendido":    ["limitado", "restringido temporalmente"],
    "blocked":       ["limited", "temporarily restricted"],
    "bloqueado":     ["limitado", "en revisión"],
    "verify":        ["review", "check", "confirm once"],
    "verificar":     ["revisar", "consultar", "confirmar"],
}

_NEGATION_WRAPPERS = [
    "No te preocupes, pero {msg}",
    "Aunque no es urgente, {msg}",
    "Sin presiones, {msg}",
    "Cuando puedas, {msg}",
]


class AdversarialGenerator:
    """
    Genera variantes adversariales de mensajes de fraude.

    Las variantes mantienen el contenido fraudulento pero reducen las señales
    obvias, forzando al detector a aprender rasgos más profundos.

    Uso:
        gen = AdversarialGenerator(random_state=42)
        df_adv = gen.generate(df_fraud, predictor, n_per_message=3)
        # df_adv contiene solo los ejemplos que lograron bajar el nivel de riesgo
    """

    def __init__(self, random_state: int = 42) -> None:
        self._rng = random.Random(random_state)
        self._perturbations = [
            self._dilute,
            self._replace_synonyms,
            self._wrap_negation,
            self._add_spacing,
            self._introduce_typos,
        ]

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def generate(
        self,
        df_fraud: pd.DataFrame,
        predictor,
        n_per_message: int = 3,
        only_successful: bool = True,
    ) -> pd.DataFrame:
        """
        Genera n_per_message variantes por cada mensaje de fraude.

        Args:
            df_fraud:        DataFrame con mensajes fraudulentos (label="fraudulent").
            predictor:       Instancia de FraudPredictor o CascadePredictor.
            n_per_message:   Número de variantes a generar por mensaje.
            only_successful: Solo retener variantes que bajaron el risk_score.

        Returns:
            DataFrame con columnas TEXT_COLUMN y LABEL_COLUMN="fraudulent".
        """
        fraud_msgs = df_fraud[
            df_fraud[LABEL_COLUMN] == "fraudulent"
        ][TEXT_COLUMN].astype(str).tolist()

        logger.info(
            f"Generando {n_per_message} adversariales × {len(fraud_msgs)} mensajes "
            f"(only_successful={only_successful})..."
        )

        rows = []
        for msg in fraud_msgs:
            try:
                original_score = self._get_score(predictor, msg)
            except Exception:
                original_score = 100.0

            variants = self.perturb(msg, n=n_per_message)
            for variant in variants:
                if only_successful:
                    try:
                        variant_score = self._get_score(predictor, variant)
                        if variant_score >= original_score:
                            continue  # no logró evadir → descartar
                    except Exception:
                        pass

                rows.append({TEXT_COLUMN: variant, LABEL_COLUMN: "fraudulent"})

        df_out = pd.DataFrame(rows)
        logger.info(f"Adversariales generados: {len(df_out)}")
        return df_out

    def perturb(self, message: str, n: int = 3) -> list[str]:
        """Aplica n perturbaciones aleatorias distintas al mensaje."""
        chosen = self._rng.sample(self._perturbations, min(n, len(self._perturbations)))
        return [p(message) for p in chosen]

    # ------------------------------------------------------------------
    # Perturbaciones individuales
    # ------------------------------------------------------------------

    def _dilute(self, text: str) -> str:
        """Envuelve el mensaje con frases legítimas."""
        prefix = self._rng.choice(_LEGIT_GREETINGS)
        suffix = self._rng.choice(_LEGIT_CLOSINGS)
        return prefix + text + suffix

    def _replace_synonyms(self, text: str) -> str:
        """Reemplaza palabras de urgencia por sinónimos menos obvios."""
        result = text
        for word, synonyms in _URGENCY_SYNONYMS.items():
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            if pattern.search(result):
                replacement = self._rng.choice(synonyms)
                result = pattern.sub(replacement, result, count=1)
        return result

    def _wrap_negation(self, text: str) -> str:
        """Añade prefijo de negación para confundir al detector de urgencia."""
        wrapper = self._rng.choice(_NEGATION_WRAPPERS)
        return wrapper.format(msg=text)

    def _add_spacing(self, text: str) -> str:
        """Inserta espacios o guiones en palabras clave para evitar coincidencia exacta."""
        all_signals = URGENCY_WORDS + CREDENTIAL_WORDS
        result = text
        candidates = [w for w in all_signals if len(w) >= 6 and w in result.lower()]
        if not candidates:
            return text
        target = self._rng.choice(candidates)
        # Insertar guion en posición aleatoria dentro de la palabra
        idx = self._rng.randint(2, len(target) - 2)
        obfuscated = target[:idx] + "-" + target[idx:]
        result = re.sub(re.escape(target), obfuscated, result, count=1, flags=re.IGNORECASE)
        return result

    def _introduce_typos(self, text: str) -> str:
        """Introduce typos controlados en 1-2 palabras de señal."""
        all_signals = URGENCY_WORDS + CREDENTIAL_WORDS
        candidates = [w for w in all_signals if len(w) >= 4 and w.lower() in text.lower()]
        if not candidates:
            return text
        target = self._rng.choice(candidates)
        typo = self._make_typo(target)
        return re.sub(re.escape(target), typo, text, count=1, flags=re.IGNORECASE)

    def _make_typo(self, word: str) -> str:
        """Genera un typo cambiando un carácter."""
        _SUBSTITUTIONS = {
            "a": "4", "e": "3", "i": "1", "o": "0", "s": "$",
            "A": "4", "E": "3", "I": "1", "O": "0", "S": "$",
        }
        chars = list(word)
        candidates = [i for i, c in enumerate(chars) if c in _SUBSTITUTIONS]
        if not candidates:
            # Duplicar una letra aleatoria
            idx = self._rng.randint(1, len(chars) - 1)
            chars.insert(idx, chars[idx])
        else:
            idx = self._rng.choice(candidates)
            chars[idx] = _SUBSTITUTIONS[chars[idx]]
        return "".join(chars)

    # ------------------------------------------------------------------
    # Evaluación interna
    # ------------------------------------------------------------------

    @staticmethod
    def _get_score(predictor, message: str) -> float:
        """Extrae el risk_score numérico del predictor."""
        result = predictor.predict(message)
        return float(result.get("risk_score", 0))
