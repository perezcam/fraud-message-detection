"""
ConversationAnalyzer — detecta subsecuencias sospechosas en conversaciones.

Flujo de análisis (4 capas):
  1. Enriquecimiento individual — analyze_risk() + FraudPredictor (si disponible)
  2. Extracción de features     — WindowFeatureExtractor sobre cada ventana
  3. Scoring de patrones        — score_fn basado en features estadísticos (no keywords)
     + ConversationWindowClassifier (RandomForest + IsolationForest) si disponible
  4. LLM conversacional         — para candidatos con score >= LLM_THRESHOLD (Mistral)

El ConversationWindowClassifier aporta el componente de IA entrenado sobre
secuencias sintéticas del dataset SMS: sus predicciones modulan el score final
de cada patrón detectado.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except ImportError:
    pass

from src.conversation.features import WindowFeatureExtractor
from src.conversation.models import ConversationReport, Message, PatternMatch
from src.conversation.patterns import PATTERNS, Pattern
from src.rules.risk import analyze_risk

# ---------------------------------------------------------------------------
# Umbrales
# ---------------------------------------------------------------------------
_WINDOW_SIZES        = [2, 3, 5, 8]
_CANDIDATE_THRESHOLD = 0.40   # score mínimo para considerar ventana
_LLM_THRESHOLD       = 0.55   # score mínimo para invocar LLM
_OVERLAP_RATIO       = 0.60   # solapamiento para fusionar ventanas del mismo patrón
_MODEL_WEIGHT        = 0.45   # peso del ConversationWindowClassifier en el score final
_PATTERN_WEIGHT      = 0.55   # peso del score de patrón estadístico

_RISK_LEVELS = [
    (0.80, "critical"),
    (0.60, "high"),
    (0.35, "medium"),
    (0.00, "low"),
]

# ---------------------------------------------------------------------------
# Prompts LLM
# ---------------------------------------------------------------------------
_LLM_SYSTEM = """\
Eres un experto en ciberseguridad especializado en análisis de conversaciones fraudulentas.

Se te proporciona una subsecuencia de mensajes de una conversación. Analiza la DINÁMICA
CONVERSACIONAL (el flujo y progresión, no cada mensaje de forma aislada) y determina si
la secuencia exhibe el patrón conductual indicado.

Responde ÚNICAMENTE con un objeto JSON válido:
{
  "is_suspicious": true | false,
  "pattern_type": "<tipo_detectado>",
  "confidence": <número 0.0–1.0>,
  "key_message_indices": [<índices 0-based de los mensajes más relevantes>],
  "tactics": [<lista de tácticas de manipulación, en español>],
  "explanation": "<máximo 100 palabras en español explicando la dinámica sospechosa>"
}

IMPORTANTE: Evalúa el FLUJO y PROGRESIÓN de la conversación. Un mensaje inocente al inicio
puede ser parte de una táctica de construcción de confianza si los mensajes posteriores
revelan intenciones fraudulentas.
"""

_PATTERN_DESCRIPTIONS = {
    "urgency_escalation":    "La urgencia aumenta progresivamente a lo largo de los mensajes",
    "credential_harvesting": "Solicitud escalonada de credenciales, códigos o datos personales",
    "social_engineering":    "Mensajes iniciales inofensivos que derivan en solicitud fraudulenta",
    "impersonation":         "Suplantación de entidad oficial (banco, gobierno, empresa) + acción requerida",
    "prize_scam_sequence":   "Anuncio de premio seguido de solicitud de datos o pago para reclamarlo",
    "financial_coercion":    "Presión o amenaza combinada con solicitud de transferencia o pago",
    "trust_building_attack": "Conversación aparentemente inofensiva que culmina en solicitud fraudulenta",
}


def _score_to_risk(score: float) -> str:
    for threshold, level in _RISK_LEVELS:
        if score >= threshold:
            return level
    return "low"


# ---------------------------------------------------------------------------
# ConversationAnalyzer
# ---------------------------------------------------------------------------

class ConversationAnalyzer:
    """
    Detecta patrones de comportamiento sospechoso en secuencias de mensajes.

    Capas de análisis:
      1. Enriquecimiento individual (analyze_risk + FraudPredictor)
      2. Extracción de features estadísticos por ventana
      3. Score de patrones (features estadísticos + ConversationWindowClassifier)
      4. LLM para confirmación/interpretación de candidatos fuertes

    Args:
        api_key:    API key de Mistral (o MISTRAL_API_KEY en .env).
        llm_model:  Modelo Mistral (o MISTRAL_MODEL en .env).
        enable_llm: Invocar LLM para candidatos con score alto.
        enable_ml:  Enriquecer mensajes con FraudPredictor.
        cfg:        Overrides de umbrales en tiempo de ejecución. Claves válidas:
                    candidate_threshold, llm_threshold, overlap_ratio,
                    model_weight, pattern_weight,
                    critical_threshold, high_threshold, medium_threshold.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        enable_llm: bool = True,
        enable_ml: bool = True,
        enable_aco: bool = False,
        cfg: Optional[dict] = None,
    ) -> None:
        self._cfg: dict   = cfg or {}
        self._llm_model   = llm_model or os.environ.get("MISTRAL_MODEL", "open-mistral-nemo")
        self._api_key     = api_key or os.environ.get("MISTRAL_API_KEY", "")
        self._ml          = None
        self._llm_client  = None
        self._seq_model   = None   # ConversationWindowClassifier
        self._feat_ext    = WindowFeatureExtractor()
        self._aco         = None   # ACOConversationPath

        if enable_aco:
            try:
                from src.conversation.aco_path import ACOConversationPath
                self._aco = ACOConversationPath(n_ants=30, max_iter=50)
                logger.info("ACOConversationPath activado.")
            except Exception as exc:
                logger.warning(f"ACO no disponible: {exc}")

        # Predictor ML individual
        if enable_ml:
            try:
                from src.ml.predict import FraudPredictor
                self._ml = FraudPredictor()
                logger.info("FraudPredictor cargado para enriquecimiento.")
            except Exception as exc:
                logger.warning(f"FraudPredictor no disponible: {exc}")

        # Clasificador de secuencias conversacionales (RandomForest + IsolationForest)
        try:
            from src.conversation.sequence_model import ConversationWindowClassifier, NEURAL_FILE
            from src.config import MODELS_DIR
            clf = ConversationWindowClassifier()
            clf.load(MODELS_DIR / NEURAL_FILE)
            self._seq_model = clf
            logger.info("ConversationWindowClassifier cargado.")
        except FileNotFoundError:
            logger.info(
                "Modelo conversacional no encontrado. "
                "Para activarlo: python main.py train-conversation-model "
                "--dataset data/processed/messages.csv"
            )
        except Exception as exc:
            logger.warning(f"ConversationWindowClassifier no disponible: {exc}")

        # LLM Mistral
        if enable_llm and self._api_key:
            try:
                from mistralai.client.sdk import Mistral
                self._llm_client = Mistral(api_key=self._api_key)
                logger.info(f"LLM conversacional activo: {self._llm_model}")
            except Exception as exc:
                logger.warning(f"LLM conversacional no disponible: {exc}")

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def analyze(self, messages: list[Message]) -> ConversationReport:
        """
        Analiza una conversación completa en busca de patrones fraudulentos.

        Args:
            messages: Lista de Message en orden cronológico.

        Returns:
            ConversationReport con todas las subsecuencias sospechosas.
        """
        if not messages:
            return ConversationReport(
                total_messages=0, messages=[],
                pattern_matches=[], overall_risk="low", overall_score=0.0,
            )

        logger.info(f"Analizando conversación: {len(messages)} mensajes.")

        # Capa 1: Enriquecer
        enriched = self._enrich(messages)

        # Capa 2 + 3: Ventanas → features → scoring
        candidates = self._find_candidates(enriched)

        # Fusionar solapamientos por tipo de patrón
        overlap_ratio = self._cfg.get("overlap_ratio", _OVERLAP_RATIO)
        merged = _merge_overlapping(candidates, overlap_ratio)
        logger.info(
            f"Candidatos: {len(candidates)} → {len(merged)} tras fusión de solapamientos."
        )

        # Capa 4: LLM para candidatos fuertes
        llm_thr = self._cfg.get("llm_threshold", _LLM_THRESHOLD)
        matches: list[PatternMatch] = []
        for start, end, pattern, score, attn_weights in merged:
            window = enriched[start : end + 1]
            llm_result = None
            if score >= llm_thr and self._llm_client:
                llm_result = self._llm_analyze(window, pattern, start)

            confidence = _combine_scores(score, llm_result)
            matches.append(PatternMatch(
                pattern_type=pattern.name,
                pattern_description=pattern.description,
                start_idx=start,
                end_idx=end,
                messages=window,
                rule_score=score,
                confidence=confidence,
                risk_level=self._score_to_risk(confidence),
                llm_analysis=llm_result,
                neural_attention=attn_weights,
            ))

        matches.sort(key=lambda m: m.confidence, reverse=True)

        # Resumen LLM de la conversación completa
        llm_summary = None
        if matches and self._llm_client:
            llm_summary = self._llm_summary(enriched, matches)

        overall_score = max((m.confidence for m in matches), default=0.0)
        overall_risk  = self._score_to_risk(overall_score)

        layers = ["rules+features", "ml"]
        if self._seq_model:
            layers.append("sequence_model")
        if any(m.llm_analysis for m in matches):
            layers.append("llm")

        # Capa ACO — arco narrativo de manipulación
        aco_result = None
        if self._aco is not None:
            try:
                aco_result = self._aco.analyze(enriched, predictor=self._ml)
                layers.append("aco")
                logger.info(
                    f"ACO: escalation_start={aco_result['escalation_start']}, "
                    f"path_score={aco_result['path_score']:.2f}"
                )
            except Exception as exc:
                logger.warning(f"Error en ACO: {exc}")

        logger.info(
            f"Completado: {len(matches)} patrones, riesgo={overall_risk} "
            f"({overall_score:.0%}) | capas={layers}"
        )

        return ConversationReport(
            total_messages=len(enriched),
            messages=enriched,
            pattern_matches=matches,
            overall_risk=overall_risk,
            overall_score=overall_score,
            llm_summary=llm_summary,
            aco_analysis=aco_result,
        )

    # ------------------------------------------------------------------
    # Helpers cfg-aware
    # ------------------------------------------------------------------

    def _score_to_risk(self, score: float) -> str:
        critical_thr = self._cfg.get("critical_threshold", 0.80)
        high_thr     = self._cfg.get("high_threshold",     0.60)
        medium_thr   = self._cfg.get("medium_threshold",   0.35)
        if score >= critical_thr:
            return "critical"
        if score >= high_thr:
            return "high"
        if score >= medium_thr:
            return "medium"
        return "low"

    # ------------------------------------------------------------------
    # Enriquecimiento individual
    # ------------------------------------------------------------------

    def _enrich(self, messages: list[Message]) -> list[Message]:
        enriched = []
        for msg in messages:
            m = Message(text=msg.text, sender=msg.sender, timestamp=msg.timestamp)
            m.individual_risk = analyze_risk(msg.text)
            if self._ml:
                try:
                    ml = self._ml.predict(msg.text)
                    m.individual_risk["ml_label"]      = ml.get("predicted_class")
                    m.individual_risk["ml_confidence"] = ml.get("confidence")
                except Exception:
                    pass
            enriched.append(m)
        return enriched

    # ------------------------------------------------------------------
    # Detección por ventanas: features → score
    # ------------------------------------------------------------------

    def _find_candidates(
        self, messages: list[Message]
    ) -> list[tuple[int, int, Pattern, float]]:
        """
        Para cada ventana y cada patrón:
          1. Extrae features estadísticos (WindowFeatureExtractor)
          2. Calcula pattern_score (score_fn basado en features)
          3. Si hay ConversationWindowClassifier, obtiene model_score
          4. Score final = _PATTERN_WEIGHT * pattern_score + _MODEL_WEIGHT * model_score
        """
        n = len(messages)
        candidates: list[tuple[int, int, Pattern, float]] = []

        candidate_thr = self._cfg.get("candidate_threshold", _CANDIDATE_THRESHOLD)
        pattern_w     = self._cfg.get("pattern_weight",      _PATTERN_WEIGHT)
        model_w       = self._cfg.get("model_weight",        _MODEL_WEIGHT)

        for window_size in _WINDOW_SIZES:
            if window_size > n:
                continue
            for start in range(n - window_size + 1):
                end    = start + window_size - 1
                window = messages[start : end + 1]

                # Extraer features una sola vez por ventana
                features = self._feat_ext.extract(window)

                # Obtener model_score del BiLSTM (pasa mensajes directamente)
                model_score  = -1.0
                attn_weights = []
                if self._seq_model:
                    model_score, attn_weights = (
                        self._seq_model.predict_with_attention(window)
                    )

                for pattern in PATTERNS:
                    if window_size < pattern.min_window_size:
                        continue

                    pattern_score = pattern.score_fn(features)

                    if model_score >= 0:
                        score = pattern_w * pattern_score + model_w * model_score
                    else:
                        score = pattern_score

                    if score >= candidate_thr:
                        candidates.append(
                            (start, end, pattern, score, attn_weights)
                        )

        return candidates

    # ------------------------------------------------------------------
    # LLM — análisis de subsecuencia
    # ------------------------------------------------------------------

    def _llm_analyze(
        self, window: list[Message], pattern: Pattern, start_idx: int
    ) -> Optional[dict]:
        user_content = _format_window_for_llm(window, pattern, start_idx)
        try:
            resp = self._llm_client.chat.complete(
                model=self._llm_model,
                messages=[
                    {"role": "system", "content": _LLM_SYSTEM},
                    {"role": "user",   "content": user_content},
                ],
                response_format={"type": "json_object"},
                max_tokens=500,
                temperature=0.05,
            )
            result = json.loads(resp.choices[0].message.content)
            result.setdefault("is_suspicious",       False)
            result.setdefault("pattern_type",        pattern.name)
            result.setdefault("confidence",          0.5)
            result.setdefault("key_message_indices", [])
            result.setdefault("tactics",             [])
            result.setdefault("explanation",         "")
            return result
        except Exception as exc:
            logger.error(f"Error en LLM conversacional: {exc}")
            return None

    # ------------------------------------------------------------------
    # LLM — resumen narrativo
    # ------------------------------------------------------------------

    def _llm_summary(
        self, messages: list[Message], matches: list[PatternMatch]
    ) -> Optional[str]:
        try:
            prompt = _build_summary_prompt(messages, matches)
            resp = self._llm_client.chat.complete(
                model=self._llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Eres un experto en seguridad. Resume brevemente (máx 150 palabras) "
                            "los patrones de fraude detectados en la conversación. Responde en "
                            "español, en texto plano, sin formato markdown ni viñetas."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=350,
                temperature=0.1,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            logger.error(f"Error generando resumen: {exc}")
            return None


# ---------------------------------------------------------------------------
# Funciones puras auxiliares
# ---------------------------------------------------------------------------

def _merge_overlapping(
    candidates: list[tuple],
    overlap_ratio: float = _OVERLAP_RATIO,
) -> list[tuple]:
    """
    Fusiona ventanas solapadas del MISMO tipo de patrón, conservando la de
    mayor score. Patrones distintos se mantienen separados.
    Maneja tuplas de 5 elementos: (start, end, pattern, score, attn_weights).
    """
    if not candidates:
        return []

    by_pattern: dict[str, list[tuple]] = {}
    for item in candidates:
        by_pattern.setdefault(item[2].name, []).append(item)

    merged_all: list[tuple] = []
    for group in by_pattern.values():
        sorted_g = sorted(group, key=lambda x: (x[0], -x[3]))
        merged: list[tuple] = []
        for item in sorted_g:
            start, end, pattern, score = item[0], item[1], item[2], item[3]
            absorbed = False
            for i, prev in enumerate(merged):
                ms, me, mscore = prev[0], prev[1], prev[3]
                overlap = max(0, min(end, me) - max(start, ms) + 1)
                min_len = min(end - start + 1, me - ms + 1)
                if min_len > 0 and overlap / min_len >= overlap_ratio:
                    if score > mscore:
                        merged[i] = item
                    absorbed = True
                    break
            if not absorbed:
                merged.append(item)
        merged_all.extend(merged)

    return merged_all


def _combine_scores(score: float, llm_result: Optional[dict]) -> float:
    """Combina el score de reglas+ML con la confianza del LLM si lo confirmó."""
    if not llm_result or not llm_result.get("is_suspicious", False):
        return score
    llm_conf = float(llm_result.get("confidence", 0.5))
    w_llm = 0.60 if llm_conf >= 0.70 else 0.40
    return round(w_llm * llm_conf + (1 - w_llm) * score, 3)


def _format_window_for_llm(
    messages: list[Message], pattern: Pattern, start_idx: int
) -> str:
    lines = [
        f"Subsecuencia [{start_idx}–{start_idx + len(messages) - 1}] "
        f"({len(messages)} mensajes):",
        "─" * 60,
    ]
    for i, msg in enumerate(messages):
        risk_note = ""
        if msg.individual_risk:
            rs = msg.individual_risk.get("risk_score", 0)
            if rs >= 25:
                risk_note = f"  ⚠ risk={rs}/100"
        lines.append(f"  [{i}] ({msg.sender}) \"{msg.text}\"{risk_note}")
    lines.append("─" * 60)
    desc = _PATTERN_DESCRIPTIONS.get(pattern.name, pattern.description)
    lines.append(f"\nPatrón detectado: {pattern.name}")
    lines.append(f"Descripción: {desc}")

    high_risk = [
        (i, m) for i, m in enumerate(messages)
        if m.individual_risk and m.individual_risk.get("risk_score", 0) >= 25
    ]
    if high_risk:
        lines.append("\nSeñales individuales:")
        for i, m in high_risk:
            sigs = m.individual_risk.get("signals", [])[:3]
            if sigs:
                lines.append(f"  [{i}]: {', '.join(sigs)}")
    return "\n".join(lines)


def _build_summary_prompt(messages: list[Message], matches: list[PatternMatch]) -> str:
    lines = [f"Conversación de {len(messages)} mensajes.", ""]
    lines.append(f"Patrones detectados ({len(matches)}):")
    for m in matches[:5]:
        lines.append(
            f"  • {m.pattern_type} {m.span} "
            f"(confianza={m.confidence:.0%}): {m.pattern_description}"
        )
        if m.llm_analysis and m.llm_analysis.get("tactics"):
            tacs = ", ".join(m.llm_analysis["tactics"][:3])
            lines.append(f"    Tácticas: {tacs}")
    return "\n".join(lines)
