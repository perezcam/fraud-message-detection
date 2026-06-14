"""
Modelos de datos para el análisis de conversaciones.

Message            — entrada: un mensaje individual con metadatos opcionales
PatternMatch       — salida: una subsecuencia sospechosa detectada
ConversationReport — salida: reporte completo del análisis de la conversación
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Message:
    """Un mensaje individual de la conversación."""

    text: str
    sender: str = "unknown"         # "suspect" | "user" | nombre libre
    timestamp: Optional[str] = None

    # Llenado durante el análisis (no es parte de la entrada pública)
    individual_risk: Optional[dict] = field(default=None, repr=False)


@dataclass
class PatternMatch:
    """Una subsecuencia sospechosa detectada en la conversación."""

    pattern_type: str
    pattern_description: str
    start_idx: int
    end_idx: int
    messages: list[Message]        # mensajes de la subsecuencia
    rule_score: float               # puntuación del motor de reglas (0–1)
    confidence: float               # confianza final combinada (0–1)
    risk_level: str                 # "low" | "medium" | "high" | "critical"
    llm_analysis: Optional[dict] = None       # análisis del LLM si fue invocado
    neural_attention: Optional[list] = None   # attention weights del BiLSTM por mensaje

    @property
    def span(self) -> str:
        return f"[{self.start_idx}–{self.end_idx}]"


@dataclass
class ConversationReport:
    """Resultado del análisis de una conversación completa."""

    total_messages: int
    messages: list[Message]                 # mensajes enriquecidos
    pattern_matches: list[PatternMatch]     # subsecuencias sospechosas
    overall_risk: str                       # "low" | "medium" | "high" | "critical"
    overall_score: float                    # máximo de las confianzas individuales
    llm_summary:  Optional[str]  = None    # resumen narrativo del LLM
    aco_analysis: Optional[dict] = None    # análisis ACO del arco de manipulación

    def has_suspicious_patterns(self) -> bool:
        return bool(self.pattern_matches)

    def to_dict(self) -> dict:
        return {
            "total_messages": self.total_messages,
            "overall_risk": self.overall_risk,
            "overall_score": round(self.overall_score, 3),
            "patterns_found": len(self.pattern_matches),
            "pattern_matches": [
                {
                    "pattern_type": m.pattern_type,
                    "description": m.pattern_description,
                    "span": m.span,
                    "messages": [
                        {"idx": m.start_idx + i, "text": msg.text, "sender": msg.sender}
                        for i, msg in enumerate(m.messages)
                    ],
                    "rule_score": round(m.rule_score, 3),
                    "confidence": round(m.confidence, 3),
                    "risk_level": m.risk_level,
                    "llm_analysis": m.llm_analysis,
                    "neural_attention": (
                        [round(w, 4) for w in m.neural_attention]
                        if m.neural_attention else None
                    ),
                }
                for m in self.pattern_matches
            ],
            "per_message_risk": [
                {
                    "idx": i,
                    "text": msg.text[:80],
                    "sender": msg.sender,
                    "risk_score": msg.individual_risk.get("risk_score", 0) if msg.individual_risk else 0,
                    "risk_level": msg.individual_risk.get("risk_level", "low") if msg.individual_risk else "low",
                    "signals": msg.individual_risk.get("signals", []) if msg.individual_risk else [],
                }
                for i, msg in enumerate(self.messages)
            ],
            "llm_summary":  self.llm_summary,
            "aco_analysis": self.aco_analysis,
        }
