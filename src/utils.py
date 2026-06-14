"""
Utilidades generales del proyecto.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any


def setup_logging(level: int = logging.INFO, log_file: Path | None = None) -> None:
    """Configura el logger raíz con salida a consola y opcionalmente a archivo."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def load_json(path: Path) -> Any:
    """Carga un archivo JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path, indent: int = 2) -> None:
    """Guarda datos como archivo JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str, ensure_ascii=False)


def print_conversation_report(report) -> None:
    """Imprime el ConversationReport de forma legible en consola."""
    RISK_ICONS  = {"low": "🟢", "medium": "🟡", "high": "🔴", "critical": "🚨"}
    RISK_LABELS = {"low": "BAJO", "medium": "MEDIO", "high": "ALTO", "critical": "CRÍTICO"}

    icon  = RISK_ICONS.get(report.overall_risk, "⚪")
    label = RISK_LABELS.get(report.overall_risk, report.overall_risk.upper())

    print("\n" + "=" * 65)
    print("  ANÁLISIS DE CONVERSACIÓN")
    print("=" * 65)
    print(f"  Mensajes analizados : {report.total_messages}")
    print(f"  Riesgo general      : {icon} {label}")
    print(f"  Score general       : {report.overall_score:.0%}")
    print(f"  Patrones detectados : {len(report.pattern_matches)}")

    if report.pattern_matches:
        print()
        for i, m in enumerate(report.pattern_matches, 1):
            m_icon  = RISK_ICONS.get(m.risk_level, "⚪")
            m_label = RISK_LABELS.get(m.risk_level, m.risk_level.upper())
            print(f"  [{i}] {m.pattern_type}  {m.span}")
            print(f"      {m.pattern_description}")
            print(f"      Confianza: {m.confidence:.0%}  |  Riesgo: {m_icon} {m_label}")
            if m.llm_analysis and m.llm_analysis.get("explanation"):
                print(f"      LLM: {m.llm_analysis['explanation']}")
            if m.llm_analysis and m.llm_analysis.get("tactics"):
                tacs = ", ".join(m.llm_analysis["tactics"][:3])
                print(f"      Tácticas: {tacs}")
            if m.neural_attention:
                # Mostrar qué mensaje fue más determinante según el BiLSTM
                max_i   = int(max(range(len(m.neural_attention)), key=lambda i: m.neural_attention[i]))
                max_w   = m.neural_attention[max_i]
                abs_idx = m.start_idx + max_i
                print(f"      Atención BiLSTM: msg[{abs_idx}] = {max_w:.2%} (más determinante)")
    else:
        print()
        print("  No se detectaron patrones sospechosos.")

    if report.llm_summary:
        print()
        print("  Resumen del análisis:")
        for line in report.llm_summary.split("\n"):
            print(f"    {line}")

    print("=" * 65 + "\n")


def print_prediction(result: dict) -> None:
    """Imprime el resultado de una predicción de forma legible en consola."""
    RISK_ICONS = {"low": "🟢", "medium": "🟡", "high": "🔴"}
    RISK_LABELS = {"low": "BAJO", "medium": "MEDIO", "high": "ALTO"}

    icon = RISK_ICONS.get(result["risk_level"], "⚪")
    label = RISK_LABELS.get(result["risk_level"], result["risk_level"].upper())

    print("\n" + "=" * 65)
    print("  ANÁLISIS DE MENSAJE FRAUDULENTO")
    print("=" * 65)
    print(f"  Mensaje       : {result['original_message'][:75]}")
    if len(result["original_message"]) > 75:
        print(f"                  ...{result['original_message'][75:125]}")
    print(f"  Clase         : {result['predicted_class'].upper()}")
    print(f"  Nivel riesgo  : {icon} {label}")
    if result.get("confidence") is not None:
        print(f"  Confianza     : {result['confidence']:.1%}")
    print(f"  Puntaje reglas: {result.get('risk_score', 'N/A')}/100")
    print()
    if result["signals"]:
        print("  Señales detectadas:")
        for signal in result["signals"]:
            print(f"    • {signal}")
    else:
        print("  Señales detectadas: Ninguna")
    print()
    print(f"  Recomendación:")
    print(f"    {result['recommendation']}")
    print("=" * 65 + "\n")
