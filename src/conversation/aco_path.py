"""
Detección del camino de manipulación en conversaciones usando Colonia de Hormigas (ACO).

Fuente: Metaheurísticas (INFOS/4- Metaheuristicas.pdf) — Colonia de Hormigas.

Modela la conversación como un grafo dirigido:
  - Nodos   = mensajes individuales de la conversación
  - Arcos   = transiciones entre mensajes consecutivos
  - Peso heurístico h[i][j] = risk_score del mensaje j (cómo de sospechoso es ir allí)
  - Feromonas τ[i][j] = señal aprendida: arcos que las hormigas prefieren

Las hormigas construyen caminos de longitud L siguiendo:
  P(i→j) ∝ τ[i][j]^α × h[j]^β

Cuando una hormiga encuentra un buen camino (muchos mensajes sospechosos):
  → deposita más feromonas (Δτ = Q / (1 - calidad))

Los arcos sin visitar evaporan: τ[i][j] *= (1 - ρ)

Tras max_iter iteraciones, el camino con más feromonas acumuladas es
el "arco narrativo del fraude": la secuencia de mensajes donde el estafador
va escalando la manipulación.

Aportación única: no solo dice SI hay fraude, sino EN QUÉ MOMENTO empieza
y cuál es la cadena de mensajes más sospechosa.

Uso:
    from src.conversation.models import Message
    aco = ACOConversationPath(n_ants=30, max_iter=50)
    result = aco.analyze(messages, predictor=None)
    print(result["manipulation_arc"])
    print(result["escalation_start"])
"""

import logging
import random
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ACOConversationPath:
    """
    Análisis de conversaciones con Colonia de Hormigas para detectar el
    arco narrativo de manipulación.

    Uso:
        aco = ACOConversationPath(n_ants=30, max_iter=50)
        result = aco.analyze(messages, predictor)
        print(result["worst_path"])       # [1, 3, 5] — índices de mensajes
        print(result["escalation_start"]) # 1 — donde empieza la manipulación
        print(result["manipulation_arc"]) # descripción legible
    """

    def __init__(
        self,
        n_ants:       int   = 30,
        max_iter:     int   = 50,
        alpha:        float = 1.0,
        beta:         float = 2.0,
        rho:          float = 0.1,
        q:            float = 1.0,
        path_length:  int   = 4,
        random_state: int   = 42,
    ) -> None:
        self.n_ants      = n_ants
        self.max_iter    = max_iter
        self.alpha       = alpha
        self.beta        = beta
        self.rho         = rho
        self.q           = q
        self.path_length = path_length
        self._rng        = random.Random(random_state)
        self._np_rng     = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def analyze(self, messages, predictor=None) -> dict:
        """
        Encuentra el camino de manipulación más sospechoso en la conversación.

        Args:
            messages:  list[Message] — conversación completa.
            predictor: FraudPredictor opcional. Si es None, usa heurísticas de reglas.

        Returns:
            {
                "worst_path":       list[int],   — índices de mensajes del camino,
                "worst_path_texts": list[str],
                "path_score":       float,        — 0-1 severidad,
                "escalation_start": int,          — índice donde empieza la escalada,
                "pheromone_map":    list[list],   — matriz τ final,
                "convergence":      list[float],  — best path_score por iteración,
                "manipulation_arc": str,          — descripción legible,
            }
        """
        n = len(messages)

        # Caso trivial
        if n == 0:
            return self._empty_result()
        if n == 1:
            score = self._get_risk(messages[0], predictor)
            return {
                "worst_path":       [0],
                "worst_path_texts": [getattr(messages[0], "text", str(messages[0]))],
                "path_score":       round(score, 4),
                "escalation_start": 0,
                "pheromone_map":    [[1.0]],
                "convergence":      [round(score, 4)] * self.max_iter,
                "manipulation_arc": self._describe([0], [score], messages),
            }

        # Obtener risk_score normalizado [0-1] de cada mensaje
        risk = self._get_all_risks(messages, predictor)

        # Inicializar feromonas
        tau = np.ones((n, n), dtype=float)
        np.fill_diagonal(tau, 0.0)  # sin auto-loops

        # Heurística h[j] = risk del nodo j (destino)
        h = np.array(risk, dtype=float)

        L         = min(self.path_length, n)
        best_path = list(range(min(L, n)))
        best_score = float(np.mean([risk[i] for i in best_path]))
        convergence: list[float] = []

        for _it in range(self.max_iter):
            paths    = []
            scores   = []
            delta_tau = np.zeros((n, n), dtype=float)

            for _ in range(self.n_ants):
                path  = self._build_path(tau, h, n, L)
                score = float(np.mean([risk[idx] for idx in path]))
                paths.append(path)
                scores.append(score)

                # Depositar feromonas proporcionales a la calidad del camino
                deposit = self.q / max(1.0 - score + 1e-6, 1e-6)
                for k in range(len(path) - 1):
                    delta_tau[path[k], path[k + 1]] += deposit

            # Evaporación + depósito
            tau = (1.0 - self.rho) * tau + delta_tau
            np.fill_diagonal(tau, 0.0)

            # Mejor de esta iteración
            best_it_idx = int(np.argmax(scores))
            if scores[best_it_idx] > best_score:
                best_score = scores[best_it_idx]
                best_path  = paths[best_it_idx]

            convergence.append(round(best_score, 4))

        worst_texts = [getattr(messages[i], "text", str(messages[i]))
                       for i in best_path]
        path_risks  = [risk[i] for i in best_path]
        escalation  = self._find_escalation_start(best_path, risk)
        arc_desc    = self._describe(best_path, path_risks, messages)

        return {
            "worst_path":       best_path,
            "worst_path_texts": worst_texts,
            "path_score":       round(best_score, 4),
            "escalation_start": escalation,
            "pheromone_map":    tau.round(3).tolist(),
            "convergence":      convergence,
            "manipulation_arc": arc_desc,
        }

    # ------------------------------------------------------------------
    # Construcción de camino (hormiga individual)
    # ------------------------------------------------------------------

    def _build_path(self, tau: np.ndarray, h: np.ndarray, n: int, L: int) -> list[int]:
        """Una hormiga construye un camino de longitud L desde un nodo inicial aleatorio."""
        start = self._rng.randint(0, n - 1)
        path  = [start]
        visited = {start}

        for _ in range(L - 1):
            current = path[-1]
            # Candidatos: nodos no visitados
            candidates = [j for j in range(n) if j not in visited]
            if not candidates:
                break

            # Probabilidad de ir de current a j
            scores = np.array([
                (tau[current, j] ** self.alpha) * (max(h[j], 1e-6) ** self.beta)
                for j in candidates
            ], dtype=float)

            total = scores.sum()
            if total <= 0:
                next_node = self._rng.choice(candidates)
            else:
                probs     = scores / total
                next_node = candidates[self._np_rng.choice(len(candidates), p=probs)]

            path.append(next_node)
            visited.add(next_node)

        return path

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def _get_all_risks(self, messages, predictor) -> list[float]:
        """Obtiene risk_score [0-1] para cada mensaje."""
        risks = []
        for msg in messages:
            risks.append(self._get_risk(msg, predictor))
        return risks

    def _get_risk(self, message, predictor) -> float:
        """Risk score normalizado [0-1] de un mensaje."""
        text = getattr(message, "text", str(message))

        if predictor is not None:
            try:
                result = predictor.predict(text)
                return float(result.get("risk_score", 0)) / 100.0
            except Exception:
                pass

        # Fallback: heurística de señales (sin predictor)
        return self._heuristic_risk(text)

    @staticmethod
    def _heuristic_risk(text: str) -> float:
        """Heurística simple sin predictor: cuenta señales de fraude."""
        from src.config import URGENCY_WORDS, CREDENTIAL_WORDS, MONEY_WORDS
        import re
        tl = text.lower()
        score = 0.0
        score += sum(0.15 for w in URGENCY_WORDS    if w in tl)
        score += sum(0.20 for w in CREDENTIAL_WORDS if w in tl)
        score += sum(0.10 for w in MONEY_WORDS      if w in tl)
        if re.search(r"https?://|www\.", tl):
            score += 0.25
        if re.search(r"\b\d{7,}\b", tl):
            score += 0.10
        return min(score, 1.0)

    def _find_escalation_start(self, path: list[int], risk: list[float]) -> int:
        """Índice del primer mensaje del camino que supera el umbral de riesgo medio."""
        if not path:
            return 0
        mean_risk = np.mean(risk)
        threshold = max(mean_risk, 0.3)
        for idx in path:
            if risk[idx] >= threshold:
                return idx
        return path[0]

    def _describe(self, path: list[int], path_risks: list[float], messages) -> str:
        """Genera descripción legible del arco de manipulación."""
        if not path:
            return "Sin patrón de manipulación detectado."

        parts = []
        for rank, (idx, r) in enumerate(zip(path, path_risks)):
            text = getattr(messages[idx], "text", str(messages[idx]))
            snippet = text[:50].replace("\n", " ") + ("…" if len(text) > 50 else "")
            level = "baja" if r < 0.3 else "media" if r < 0.6 else "alta"
            parts.append(f"msg[{idx}] (riesgo {level}: {r:.0%}) «{snippet}»")

        score = float(np.mean(path_risks))
        if score >= 0.7:
            label = "ESCALADA SEVERA"
        elif score >= 0.4:
            label = "ESCALADA MODERADA"
        else:
            label = "PATRÓN LEVE"

        return f"{label}: " + " → ".join(parts)

    @staticmethod
    def _empty_result() -> dict:
        return {
            "worst_path":       [],
            "worst_path_texts": [],
            "path_score":       0.0,
            "escalation_start": 0,
            "pheromone_map":    [],
            "convergence":      [],
            "manipulation_arc": "Sin mensajes para analizar.",
        }
