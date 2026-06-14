"""
Análisis de robustez Monte Carlo para el detector de fraude.

Fuente: Metaheurísticas (INFOS/4- Metaheuristicas.pdf) — Random Search y criterios de parada.

Idea: si la predicción es robusta, 200 variantes ligeramente perturbadas del mismo
mensaje deberían dar resultados similares. Si el score varía mucho (alta desviación
estándar), la predicción es frágil y podría cambiar con pequeñas modificaciones del texto.

Métricas clave:
  stability = 1 - (std / max(mean_score, 1))   → 1.0 = predicción invariante
  fraud_rate = fracción de variantes predichas como fraudulentas

Uso:
    mc = MonteCarloAnalyzer(n_simulations=200)
    result = mc.analyze("ALERTA BBVA: verifique su cuenta", predictor)
    # result["stability"] → 0.87 (muy estable)
    # result["fraud_rate"] → 0.94 (94% de variantes = fraude)
"""

import logging
import random
import re
from typing import Optional

import numpy as np

from src.ml.adversarial import _URGENCY_SYNONYMS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_FILLER_WORDS = [
    "bien", "favor", "así", "solo", "más", "muy", "también",
    "siempre", "ahí", "aquí", "esto", "eso", "todo", "nada",
]


class MonteCarloAnalyzer:
    """
    Análisis de robustez mediante simulación Monte Carlo.

    Para cada mensaje aplica n_simulations perturbaciones aleatorias y mide
    cuánto varía la predicción del detector — respondiendo si la clasificación
    es estable o dependiente de palabras específicas.

    Perturbaciones implementadas (5 tipos):
      1. swap_chars     — intercambia 2 caracteres en una palabra
      2. delete_word    — elimina una palabra de relleno
      3. insert_filler  — inserta una palabra inocua en posición aleatoria
      4. change_case    — alterna mayúsculas/minúsculas de una palabra
      5. synonym_swap   — reemplaza palabra de urgencia por sinónimo
    """

    def __init__(
        self,
        n_simulations: int = 200,
        random_state:  int = 42,
    ) -> None:
        self.n_simulations = n_simulations
        self._rng = random.Random(random_state)
        self._perturbation_fns = [
            self._swap_chars,
            self._delete_word,
            self._insert_filler,
            self._change_case,
            self._synonym_swap,
        ]

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def analyze(self, message: str, predictor) -> dict:
        """
        Ejecuta n_simulations perturbaciones y calcula estadísticas de robustez.

        Args:
            message:   Texto original a analizar.
            predictor: Cualquier objeto con método predict(str) → dict con "risk_score"
                       y "predicted_class".

        Returns:
            {
                "mean_score":    float,   — media del risk_score [0-100]
                "std_score":     float,   — desviación estándar
                "ci_low":        float,   — percentil 5
                "ci_high":       float,   — percentil 95
                "stability":     float,   — 1.0 = completamente estable [0-1]
                "fraud_rate":    float,   — fracción de variantes = fraude [0-1]
                "original_score":float,   — score del mensaje sin perturbar
                "n_simulations": int,
                "verdict":       str,     — descripción legible de la estabilidad
            }
        """
        # Score del mensaje original
        try:
            orig_result = predictor.predict(message)
            original_score = float(orig_result.get("risk_score", 0))
        except Exception as exc:
            logger.warning(f"Error en predicción original: {exc}")
            original_score = 0.0

        # Simulaciones
        scores: list[float] = []
        fraud_count = 0

        for _ in range(self.n_simulations):
            variant = self._perturb(message)
            try:
                result = predictor.predict(variant)
                score = float(result.get("risk_score", 0))
                scores.append(score)
                if result.get("predicted_class") == "fraudulent":
                    fraud_count += 1
            except Exception:
                scores.append(original_score)

        arr = np.array(scores, dtype=float)
        mean  = float(np.mean(arr))
        std   = float(np.std(arr))
        ci_lo = float(np.percentile(arr, 5))
        ci_hi = float(np.percentile(arr, 95))

        stability  = float(1.0 - std / max(mean, 1.0))
        stability  = max(0.0, min(1.0, stability))
        fraud_rate = float(fraud_count / max(self.n_simulations, 1))

        verdict = self._verdict(stability, fraud_rate)

        logger.info(
            f"Monte Carlo ({self.n_simulations} sim): "
            f"mean={mean:.1f}, std={std:.1f}, "
            f"stability={stability:.2f}, fraud_rate={fraud_rate:.2f}"
        )

        return {
            "mean_score":     round(mean, 2),
            "std_score":      round(std, 2),
            "ci_low":         round(ci_lo, 2),
            "ci_high":        round(ci_hi, 2),
            "stability":      round(stability, 4),
            "fraud_rate":     round(fraud_rate, 4),
            "original_score": round(original_score, 2),
            "n_simulations":  self.n_simulations,
            "verdict":        verdict,
        }

    def _perturb(self, text: str) -> str:
        """Aplica una perturbación aleatoria al texto."""
        fn = self._rng.choice(self._perturbation_fns)
        result = fn(text)
        # Garantiza que no devuelva cadena vacía
        return result if result.strip() else text

    # ------------------------------------------------------------------
    # Perturbaciones
    # ------------------------------------------------------------------

    def _swap_chars(self, text: str) -> str:
        """Intercambia 2 caracteres adyacentes en una palabra aleatoria."""
        words = text.split()
        eligible = [i for i, w in enumerate(words) if len(w) >= 3]
        if not eligible:
            return text
        idx = self._rng.choice(eligible)
        w = list(words[idx])
        pos = self._rng.randint(0, len(w) - 2)
        w[pos], w[pos + 1] = w[pos + 1], w[pos]
        words[idx] = "".join(w)
        return " ".join(words)

    def _delete_word(self, text: str) -> str:
        """Elimina una palabra corta (≤5 chars) de manera aleatoria."""
        words = text.split()
        if len(words) <= 2:
            return text
        eligible = [i for i, w in enumerate(words) if len(w) <= 5]
        if not eligible:
            eligible = list(range(len(words)))
        idx = self._rng.choice(eligible)
        words.pop(idx)
        return " ".join(words)

    def _insert_filler(self, text: str) -> str:
        """Inserta una palabra de relleno inocua en posición aleatoria."""
        words = text.split()
        pos = self._rng.randint(0, len(words))
        filler = self._rng.choice(_FILLER_WORDS)
        words.insert(pos, filler)
        return " ".join(words)

    def _change_case(self, text: str) -> str:
        """Cambia la capitalización de una palabra aleatoria."""
        words = text.split()
        if not words:
            return text
        idx = self._rng.randint(0, len(words) - 1)
        w = words[idx]
        if w.islower():
            words[idx] = w.upper()
        elif w.isupper():
            words[idx] = w.lower()
        else:
            words[idx] = w.swapcase()
        return " ".join(words)

    def _synonym_swap(self, text: str) -> str:
        """Reemplaza una palabra de urgencia por un sinónimo menos obvio."""
        for word, synonyms in _URGENCY_SYNONYMS.items():
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            if pattern.search(text):
                replacement = self._rng.choice(synonyms)
                return pattern.sub(replacement, text, count=1)
        return text

    # ------------------------------------------------------------------
    # Descripción del veredicto
    # ------------------------------------------------------------------

    @staticmethod
    def _verdict(stability: float, fraud_rate: float) -> str:
        if stability >= 0.85 and fraud_rate >= 0.80:
            return "PREDICCIÓN MUY ESTABLE — fraude confirmado con alta confianza"
        if stability >= 0.85 and fraud_rate < 0.30:
            return "PREDICCIÓN MUY ESTABLE — mensaje legítimo con alta confianza"
        if stability >= 0.70:
            return "PREDICCIÓN ESTABLE — resultado consistente ante variaciones"
        if stability >= 0.50:
            return "PREDICCIÓN MODERADA — algunos cambios alteran el veredicto"
        return "PREDICCIÓN INESTABLE — resultado sensible al texto exacto"
