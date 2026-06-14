"""
Optimización de umbrales del CascadePredictor con Recocido Simulado.

Fuente: Metaheurísticas (INFOS/5- Metaheurísticas más.pdf), sección Recocido Simulado.

Los umbrales de la cascada se fijaron manualmente. SA los optimiza automáticamente:
  - Temperatura alta inicial → acepta soluciones peores (exploración global)
  - Temperatura decae gradualmente → solo acepta mejoras (explotación local)
  - Función objetivo: F1-fraude sobre el dataset de validación

Vector de umbrales optimizado (5 dimensiones):
  [gate1_conf, gate2_emb_fraud, anomaly_thr, bayes_high, meta_fraud_thr]

Los umbrales óptimos se guardan en models/optimal_thresholds.json y
la cascada los carga automáticamente si el archivo existe.
"""

import json
import logging
import math
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import LABEL_COLUMN, MODELS_DIR, OPTIMAL_THRESHOLDS_FILE, TEXT_COLUMN

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valores por defecto de los umbrales (deben coincidir con cascade.py)
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS = {
    "gate1_conf":      0.95,
    "gate2_emb_fraud": 0.88,
    "anomaly_thr":     0.75,
    "bayes_high":      0.85,
    "meta_fraud_thr":  0.70,
}

# Rango válido para cada umbral
_BOUNDS = {
    "gate1_conf":      (0.70, 0.99),
    "gate2_emb_fraud": (0.60, 0.99),
    "anomaly_thr":     (0.50, 0.99),
    "bayes_high":      (0.50, 0.99),
    "meta_fraud_thr":  (0.40, 0.90),
}


class ThresholdOptimizer:
    """
    Optimiza el vector de umbrales del CascadePredictor usando Recocido Simulado.

    Algoritmo SA:
      1. Empieza con los umbrales por defecto
      2. Genera una perturbación aleatoria pequeña (vecino)
      3. Si mejora → acepta
      4. Si empeora → acepta con probabilidad exp(-ΔE / T)  (escape de mínimos locales)
      5. Enfría la temperatura: T = T * cooling_rate
      6. Repite hasta max_iter

    Uso:
        opt = ThresholdOptimizer()
        result = opt.optimize(cascade, df_val)
        opt.save(result["best_thresholds"])
    """

    def __init__(
        self,
        initial_temp:  float = 1.0,
        cooling_rate:  float = 0.95,
        max_iter:      int   = 300,
        step_size:     float = 0.05,
        random_state:  int   = 42,
    ) -> None:
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iter     = max_iter
        self.step_size    = step_size
        self._rng = random.Random(random_state)

    # ------------------------------------------------------------------
    # Optimización
    # ------------------------------------------------------------------

    def optimize(self, cascade, df_val: pd.DataFrame) -> dict:
        """
        Optimiza los umbrales usando el dataset de validación.

        Args:
            cascade: Instancia de CascadePredictor ya inicializada.
            df_val:  DataFrame con TEXT_COLUMN y LABEL_COLUMN (20% del total).

        Returns:
            {
                "best_thresholds": dict,
                "best_f1":         float,
                "initial_f1":      float,
                "history":         list[float],
                "n_accepted":      int,
            }
        """
        df_val = df_val[df_val[LABEL_COLUMN].isin(["fraudulent", "legitimate"])].copy()
        y_true = (df_val[LABEL_COLUMN] == "fraudulent").astype(int).values
        messages = df_val[TEXT_COLUMN].astype(str).tolist()

        logger.info(
            f"Iniciando Recocido Simulado: {self.max_iter} iteraciones, "
            f"T0={self.initial_temp}, cooling={self.cooling_rate}"
        )

        current  = dict(DEFAULT_THRESHOLDS)
        best     = dict(current)
        best_f1  = self._evaluate(cascade, messages, y_true, current)
        initial_f1 = best_f1
        current_f1 = best_f1
        temp     = self.initial_temp
        history  = [best_f1]
        n_accepted = 0

        for i in range(self.max_iter):
            neighbor = self._perturb(current)
            neighbor_f1 = self._evaluate(cascade, messages, y_true, neighbor)

            delta = neighbor_f1 - current_f1
            if delta > 0 or self._rng.random() < math.exp(delta / max(temp, 1e-10)):
                current    = neighbor
                current_f1 = neighbor_f1
                n_accepted += 1

                if current_f1 > best_f1:
                    best    = dict(current)
                    best_f1 = current_f1

            temp *= self.cooling_rate
            history.append(current_f1)

            if (i + 1) % 50 == 0:
                logger.info(
                    f"  Iter {i+1}/{self.max_iter} — "
                    f"T={temp:.4f}, best_f1={best_f1:.4f}, current_f1={current_f1:.4f}"
                )

        logger.info(
            f"SA completado — F1 inicial: {initial_f1:.4f} → F1 óptimo: {best_f1:.4f} "
            f"(+{best_f1 - initial_f1:.4f}), aceptados: {n_accepted}/{self.max_iter}"
        )
        return {
            "best_thresholds": best,
            "best_f1":         round(best_f1, 4),
            "initial_f1":      round(initial_f1, 4),
            "improvement":     round(best_f1 - initial_f1, 4),
            "history":         [round(f, 4) for f in history],
            "n_accepted":      n_accepted,
        }

    def _perturb(self, thresholds: dict) -> dict:
        """Genera un vecino perturbando UN umbral aleatoriamente."""
        neighbor = dict(thresholds)
        key = self._rng.choice(list(_BOUNDS.keys()))
        lo, hi = _BOUNDS[key]
        delta = self._rng.gauss(0, self.step_size)
        neighbor[key] = max(lo, min(hi, neighbor[key] + delta))
        return neighbor

    def _evaluate(self, cascade, messages: list, y_true: np.ndarray,
                  thresholds: dict) -> float:
        """
        Evalúa un vector de umbrales sobre el dataset de validación.
        Aplica los umbrales temporalmente durante la evaluación.
        """
        from sklearn.metrics import f1_score

        # Guardar umbrales originales
        orig = {
            "gate1":    cascade._ML_CONF_CERTAIN if hasattr(cascade, "_ML_CONF_CERTAIN") else 0.95,
        }

        # Aplicar umbrales temporalmente mediante monkey-patch en el módulo
        import src.detection.cascade as _casc_mod
        _orig_g1 = _casc_mod._ML_CONF_CERTAIN
        _orig_g2 = _casc_mod._EMB_FRAUD_CONFIRM
        _orig_an = _casc_mod._ANOMALY_THRESHOLD

        _casc_mod._ML_CONF_CERTAIN  = thresholds["gate1_conf"]
        _casc_mod._EMB_FRAUD_CONFIRM = thresholds["gate2_emb_fraud"]
        _casc_mod._ANOMALY_THRESHOLD = thresholds["anomaly_thr"]

        y_pred = []
        for msg in messages:
            try:
                r = cascade._ml.predict(msg)
                label = 1 if r["predicted_class"] == "fraudulent" else 0
                y_pred.append(label)
            except Exception:
                y_pred.append(0)

        # Restaurar umbrales originales
        _casc_mod._ML_CONF_CERTAIN  = _orig_g1
        _casc_mod._EMB_FRAUD_CONFIRM = _orig_g2
        _casc_mod._ANOMALY_THRESHOLD = _orig_an

        return float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, thresholds: dict, path: Optional[Path] = None) -> Path:
        out = Path(path) if path else MODELS_DIR / OPTIMAL_THRESHOLDS_FILE
        with open(out, "w", encoding="utf-8") as f:
            json.dump(thresholds, f, indent=2)
        logger.info(f"Umbrales óptimos guardados en {out}")
        return out

    @staticmethod
    def load(path: Optional[Path] = None) -> dict:
        p = Path(path) if path else MODELS_DIR / OPTIMAL_THRESHOLDS_FILE
        if not p.exists():
            return dict(DEFAULT_THRESHOLDS)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Umbrales cargados desde {p}")
        return data
