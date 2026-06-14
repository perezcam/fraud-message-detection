"""
Optimización de umbrales del CascadePredictor con Búsqueda Tabú.

Fuente: Metaheurísticas (INFOS/4- Metaheuristicas.pdf) — Búsqueda Tabú.

La Búsqueda Tabú mantiene una lista de los últimos K vectores de umbrales visitados.
En cada iteración genera N vecinos y elige el **mejor que no esté en la lista tabú**,
garantizando que el algoritmo no repise terreno ya explorado y explore regiones nuevas.

Diferencia clave vs Recocido Simulado (ya implementado en threshold_optimizer.py):
  - SA acepta peores soluciones con cierta probabilidad (escapar mínimos locales)
  - Tabú: nunca acepta soluciones ya visitadas (garantiza diversidad)
  - Tabú genera N vecinos y elige el mejor en cada paso (no es greedy puro)
  - Combinación recomendada: correr ambos y elegir el mejor resultado

Reutiliza:
  - _BOUNDS de threshold_optimizer.py (mismo vector de 5 umbrales)
  - _evaluate() de threshold_optimizer.py (misma función objetivo F1-fraude)

Uso:
    opt = TabuOptimizer(max_iter=300, tabu_tenure=15)
    result = opt.optimize(cascade, df_val)
    opt.save(result["best_thresholds"])
"""

import json
import logging
import random
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import LABEL_COLUMN, MODELS_DIR, TEXT_COLUMN
from src.ml.threshold_optimizer import (
    DEFAULT_THRESHOLDS,
    _BOUNDS,
    ThresholdOptimizer,
)

logger = logging.getLogger(__name__)

TABU_THRESHOLDS_FILE = "tabu_thresholds.json"


class TabuOptimizer:
    """
    Optimizador de umbrales del CascadePredictor usando Búsqueda Tabú.

    El algoritmo mantiene memoria de las últimas `tabu_tenure` soluciones visitadas
    y las excluye explícitamente de la búsqueda, forzando la exploración de nuevas
    regiones del espacio de umbrales.

    Uso:
        opt = TabuOptimizer(max_iter=300, tabu_tenure=15)
        result = opt.optimize(cascade, df_val)
        opt.save(result["best_thresholds"])
    """

    def __init__(
        self,
        max_iter:    int   = 300,
        tabu_tenure: int   = 15,
        n_neighbors: int   = 10,
        step_size:   float = 0.05,
        random_state: int  = 42,
    ) -> None:
        self.max_iter    = max_iter
        self.tabu_tenure = tabu_tenure
        self.n_neighbors = n_neighbors
        self.step_size   = step_size
        self._rng        = random.Random(random_state)
        # Reutiliza el evaluador del SA existente
        self._evaluator  = ThresholdOptimizer(step_size=step_size, random_state=random_state)

    # ------------------------------------------------------------------
    # Optimización
    # ------------------------------------------------------------------

    def optimize(self, cascade, df_val: pd.DataFrame) -> dict:
        """
        Optimiza los umbrales con Búsqueda Tabú.

        Args:
            cascade:  Instancia de CascadePredictor.
            df_val:   DataFrame de validación con TEXT_COLUMN y LABEL_COLUMN.

        Returns:
            {
                "best_thresholds": dict,
                "best_f1":         float,
                "initial_f1":      float,
                "improvement":     float,
                "history":         list[float],
                "n_tabu_moves":    int,   — iteraciones donde se rechazó por tabú
            }
        """
        df_val = df_val[df_val[LABEL_COLUMN].isin(["fraudulent", "legitimate"])].copy()
        messages = df_val[TEXT_COLUMN].astype(str).tolist()
        y_true   = (df_val[LABEL_COLUMN] == "fraudulent").astype(int).values

        logger.info(
            f"Búsqueda Tabú: {self.max_iter} iter, "
            f"tenure={self.tabu_tenure}, n_vecinos={self.n_neighbors}"
        )

        current    = dict(DEFAULT_THRESHOLDS)
        best       = dict(current)
        best_f1    = self._evaluator._evaluate(cascade, messages, y_true, current)
        initial_f1 = best_f1
        current_f1 = best_f1

        tabu_list  = deque(maxlen=self.tabu_tenure)
        tabu_list.append(self._to_key(current))

        history      = [best_f1]
        n_tabu_moves = 0

        for i in range(self.max_iter):
            # Genera N vecinos
            candidates = [
                self._evaluator._perturb(current)
                for _ in range(self.n_neighbors)
            ]

            # Filtra los que están en la lista tabú
            non_tabu = [c for c in candidates
                        if self._to_key(c) not in tabu_list]

            if not non_tabu:
                # Todos son tabú → acepta el mejor de todos (criterio de aspiración)
                non_tabu = candidates
                n_tabu_moves += 1

            # Evalúa todos los candidatos no-tabú y elige el mejor
            best_cand     = None
            best_cand_f1  = -1.0
            for cand in non_tabu:
                f1 = self._evaluator._evaluate(cascade, messages, y_true, cand)
                if f1 > best_cand_f1:
                    best_cand    = cand
                    best_cand_f1 = f1

            # Mueve a la mejor posición (aunque sea peor que current — Tabú acepta)
            if best_cand is not None:
                current    = best_cand
                current_f1 = best_cand_f1
                tabu_list.append(self._to_key(current))

                if current_f1 > best_f1:
                    best    = dict(current)
                    best_f1 = current_f1

            history.append(best_f1)

            if (i + 1) % 50 == 0:
                logger.info(
                    f"  Iter {i+1}/{self.max_iter} — "
                    f"best_f1={best_f1:.4f}, tabú_bloqueados={n_tabu_moves}"
                )

        improvement = round(best_f1 - initial_f1, 4)
        logger.info(
            f"Búsqueda Tabú completada — "
            f"F1 inicial: {initial_f1:.4f} → F1 óptimo: {best_f1:.4f} "
            f"(+{improvement:.4f}), movimientos tabú: {n_tabu_moves}"
        )

        return {
            "best_thresholds": best,
            "best_f1":         round(best_f1, 4),
            "initial_f1":      round(initial_f1, 4),
            "improvement":     improvement,
            "history":         [round(f, 4) for f in history],
            "n_tabu_moves":    n_tabu_moves,
        }

    def _to_key(self, thresholds: dict) -> tuple:
        """Convierte dict de umbrales a tuple hashable (3 decimales)."""
        return tuple(round(thresholds[k], 3) for k in sorted(thresholds.keys()))

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, thresholds: dict, path: Optional[Path] = None) -> Path:
        out = Path(path) if path else MODELS_DIR / TABU_THRESHOLDS_FILE
        with open(out, "w", encoding="utf-8") as f:
            json.dump(thresholds, f, indent=2)
        logger.info(f"Umbrales Tabú guardados en {out}")
        return out

    @staticmethod
    def load(path: Optional[Path] = None) -> dict:
        p = Path(path) if path else MODELS_DIR / TABU_THRESHOLDS_FILE
        if not p.exists():
            return dict(DEFAULT_THRESHOLDS)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Umbrales Tabú cargados desde {p}")
        return data
