"""
Optimización de hiperparámetros LightGBM con PSO (Particle Swarm Optimization).

Fuente: Metaheurísticas (INFOS/4- Metaheuristicas.pdf) — PSO: Enjambre de Partículas.

El espacio de búsqueda tiene 4 dimensiones (hiperparámetros de LightGBM):
  n_estimators, max_depth, learning_rate, min_child_samples

20 partículas vuelan simultáneamente por este espacio. Cada una recuerda:
  - Su mejor posición personal (pbest) — mejor F1 que encontró individualmente
  - La mejor posición global (gbest) — mejor F1 encontrado por cualquier partícula

Actualización de velocidad y posición (paso de tiempo t → t+1):
  v[t+1] = w*v[t] + c1*r1*(pbest - x[t]) + c2*r2*(gbest - x[t])
  x[t+1] = x[t] + v[t+1]   (clampado a _BOUNDS)

  w  = inercia (0.7)     — qué tan persistente es el movimiento anterior
  c1 = cognitivo (1.5)   — cuánto atrae el best personal
  c2 = social (1.5)      — cuánto atrae el best global

Los hiperparámetros óptimos se guardan en models/pso_hyperparams.json y
pueden usarse automáticamente con:
  python3 main.py train --model lightgbm --use-pso-params

Uso:
    opt = PSOOptimizer(n_particles=20, max_iter=50)
    result = opt.optimize(df)
    opt.save(result["best_params"])
"""

import json
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import LABEL_COLUMN, MODELS_DIR, RANDOM_STATE, TEXT_COLUMN

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Espacio de búsqueda y valores por defecto
# ---------------------------------------------------------------------------

_BOUNDS: dict[str, tuple[float, float]] = {
    "n_estimators":      (50.0,  500.0),
    "max_depth":         (2.0,   10.0),
    "learning_rate":     (0.01,  0.30),
    "min_child_samples": (5.0,   80.0),
}

PSO_HYPERPARAMS_FILE = "pso_hyperparams.json"

DEFAULT_PARAMS = {
    "n_estimators":      300,
    "max_depth":         6,
    "learning_rate":     0.05,
    "min_child_samples": 20,
}


class PSOOptimizer:
    """
    Optimizador PSO para hiperparámetros de LightGBM.

    Evalúa cada combinación de hiperparámetros con StratifiedKFold(3) en el
    dataset completo → F1-fraude medio como función objetivo.

    Uso:
        opt = PSOOptimizer(n_particles=20, max_iter=50)
        result = opt.optimize(df)
        opt.save(result["best_params"])

        # Aplicar al entrenar:
        params = PSOOptimizer.load()
        train(df, model_name="lightgbm", lgbm_params=params)
    """

    def __init__(
        self,
        n_particles: int   = 20,
        max_iter:    int   = 50,
        w:           float = 0.7,
        c1:          float = 1.5,
        c2:          float = 1.5,
        random_state: int  = 42,
    ) -> None:
        self.n_particles  = n_particles
        self.max_iter     = max_iter
        self.w            = w
        self.c1           = c1
        self.c2           = c2
        self._rng         = random.Random(random_state)
        self._np_rng      = np.random.default_rng(random_state)
        self._keys        = list(_BOUNDS.keys())

    # ------------------------------------------------------------------
    # Optimización
    # ------------------------------------------------------------------

    def optimize(self, df: pd.DataFrame) -> dict:
        """
        Ejecuta el algoritmo PSO sobre el dataset.

        Args:
            df: DataFrame con TEXT_COLUMN y LABEL_COLUMN.

        Returns:
            {
                "best_params":    dict   — hiperparámetros óptimos,
                "best_f1":        float,
                "initial_f1":     float  — baseline con DEFAULT_PARAMS,
                "improvement":    float,
                "convergence":    list[float]  — best_f1 por iteración,
                "n_evaluations":  int,
            }
        """
        df = df[df[LABEL_COLUMN].isin(["fraudulent", "legitimate"])].copy()
        logger.info(
            f"PSO: {self.n_particles} partículas × {self.max_iter} iteraciones "
            f"sobre {len(df)} mensajes"
        )

        # Evaluación baseline
        initial_f1 = self._evaluate_params(df, DEFAULT_PARAMS)
        logger.info(f"  Baseline F1 (params por defecto): {initial_f1:.4f}")

        # --- Inicialización ---
        dim = len(self._keys)
        lo  = np.array([_BOUNDS[k][0] for k in self._keys])
        hi  = np.array([_BOUNDS[k][1] for k in self._keys])

        # Posiciones aleatorias dentro de _BOUNDS
        X = self._np_rng.uniform(lo, hi, size=(self.n_particles, dim))
        # Velocidades iniciales pequeñas
        V = self._np_rng.uniform(-(hi - lo) * 0.1, (hi - lo) * 0.1,
                                  size=(self.n_particles, dim))

        pbest     = X.copy()
        pbest_val = np.array([self._evaluate_params(df, self._vec_to_params(p))
                               for p in pbest])

        gbest_idx  = int(np.argmax(pbest_val))
        gbest      = pbest[gbest_idx].copy()
        gbest_val  = float(pbest_val[gbest_idx])

        convergence    = [gbest_val]
        n_evaluations  = self.n_particles  # ya evaluamos la población inicial

        logger.info(f"  Mejor F1 inicial del enjambre: {gbest_val:.4f}")

        # --- Iteraciones ---
        for it in range(self.max_iter):
            r1 = self._np_rng.uniform(0, 1, size=(self.n_particles, dim))
            r2 = self._np_rng.uniform(0, 1, size=(self.n_particles, dim))

            V = (self.w * V
                 + self.c1 * r1 * (pbest - X)
                 + self.c2 * r2 * (gbest  - X))

            X = np.clip(X + V, lo, hi)

            # Evaluar nueva posición de cada partícula
            for i, pos in enumerate(X):
                val = self._evaluate_params(df, self._vec_to_params(pos))
                n_evaluations += 1
                if val > pbest_val[i]:
                    pbest[i]     = pos.copy()
                    pbest_val[i] = val
                    if val > gbest_val:
                        gbest     = pos.copy()
                        gbest_val = val

            convergence.append(gbest_val)

            if (it + 1) % 10 == 0:
                logger.info(
                    f"  Iter {it+1}/{self.max_iter} — best_f1={gbest_val:.4f}"
                )

        best_params = self._vec_to_params(gbest)
        improvement = round(gbest_val - initial_f1, 4)

        logger.info(
            f"PSO completado — F1 inicial: {initial_f1:.4f} → "
            f"F1 óptimo: {gbest_val:.4f} (+{improvement:.4f})"
        )
        logger.info(f"  Hiperparámetros óptimos: {best_params}")

        return {
            "best_params":   best_params,
            "best_f1":       round(gbest_val, 4),
            "initial_f1":    round(initial_f1, 4),
            "improvement":   improvement,
            "convergence":   [round(v, 4) for v in convergence],
            "n_evaluations": n_evaluations,
        }

    # ------------------------------------------------------------------
    # Evaluación de hiperparámetros
    # ------------------------------------------------------------------

    def _evaluate_params(self, df: pd.DataFrame, params: dict) -> float:
        """Evalúa un conjunto de hiperparámetros con cross-validation (cv=3)."""
        try:
            from lightgbm import LGBMClassifier
            from sklearn.model_selection import StratifiedKFold, cross_val_score
            from src.ml.features import (
                build_tfidf_vectorizer,
                extract_manual_features_batch,
                TFIDF_NGRAM_RANGE,
                TFIDF_MAX_FEATURES,
                TFIDF_MIN_DF,
            )
            import scipy.sparse as sp

            texts  = df[TEXT_COLUMN].astype(str).tolist()
            labels = (df[LABEL_COLUMN] == "fraudulent").astype(int).values

            vec  = build_tfidf_vectorizer(TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_MIN_DF)
            X_tf = vec.fit_transform(texts)
            X_mn = extract_manual_features_batch(texts)
            X    = sp.hstack([X_tf, sp.csr_matrix(X_mn)], format="csr")

            model = LGBMClassifier(
                n_estimators      = int(params["n_estimators"]),
                max_depth         = int(params["max_depth"]),
                learning_rate     = float(params["learning_rate"]),
                min_child_samples = int(params["min_child_samples"]),
                scale_pos_weight  = (labels == 0).sum() / max((labels == 1).sum(), 1),
                random_state      = RANDOM_STATE,
                verbose           = -1,
            )
            cv     = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
            scores = cross_val_score(model, X, labels, cv=cv,
                                     scoring="f1", error_score=0.0)
            return float(scores.mean())
        except Exception as exc:
            logger.debug(f"Error evaluando params {params}: {exc}")
            return 0.0

    # ------------------------------------------------------------------
    # Conversión vector ↔ dict
    # ------------------------------------------------------------------

    def _vec_to_params(self, vec: np.ndarray) -> dict:
        """Convierte vector numpy a dict de hiperparámetros con los tipos correctos."""
        int_keys = {"n_estimators", "max_depth", "min_child_samples"}
        return {
            k: (int(round(float(vec[i]))) if k in int_keys else round(float(vec[i]), 4))
            for i, k in enumerate(self._keys)
        }

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, params: dict, path: Optional[Path] = None) -> Path:
        out = Path(path) if path else MODELS_DIR / PSO_HYPERPARAMS_FILE
        with open(out, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)
        logger.info(f"Hiperparámetros PSO guardados en {out}")
        return out

    @staticmethod
    def load(path: Optional[Path] = None) -> dict:
        p = Path(path) if path else MODELS_DIR / PSO_HYPERPARAMS_FILE
        if not p.exists():
            logger.info("pso_hyperparams.json no encontrado — usando DEFAULT_PARAMS")
            return dict(DEFAULT_PARAMS)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Hiperparámetros PSO cargados desde {p}")
        return data
