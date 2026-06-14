"""
Módulo de análisis LLM con Mistral AI.
Capa profunda de análisis semántico para mensajes ambiguos o de alto riesgo.
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

_SYSTEM_PROMPT = """\
Eres un experto en ciberseguridad especializado en detección de fraude, \
smishing (SMS phishing) y mensajes maliciosos en español e inglés.

Analiza el mensaje del usuario y determina si es fraudulento, sospechoso o legítimo.

Responde ÚNICAMENTE con un objeto JSON válido con esta estructura exacta:
{
  "verdict": "fraudulent" | "suspicious" | "legitimate",
  "confidence": <número entre 0.0 y 1.0>,
  "fraud_type": "phishing" | "smishing" | "prize_scam" | "urgency_scam" | "credential_theft" | "financial_scam" | "spam" | "none",
  "indicators": [<lista de señales detectadas, en español>],
  "explanation": "<explicación breve en español, máximo 80 palabras>"
}

Criterios de clasificación:
- "fraudulent": suplantación de identidad, solicitud de credenciales/OTP/PIN, URLs engañosas, \
premios falsos, amenazas de bloqueo de cuenta, urgencia extrema para actuar.
- "suspicious": tono inusual, ofertas exageradas, solicitud de datos sin contexto claro, \
remitente desconocido con petición extraña.
- "legitimate": mensaje cotidiano, comunicación esperada, sin señales de manipulación.

Si recibes contexto adicional del clasificador ML y sistema de reglas, úsalo como referencia \
pero forma tu propio juicio semántico independiente.
"""


class MistralFraudAnalyzer:
    """
    Analizador de fraude basado en Mistral AI.
    Realiza análisis semántico profundo para mensajes ambiguos o borderline.
    """

    DEFAULT_MODEL = "open-mistral-nemo"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        key = api_key or os.environ.get("MISTRAL_API_KEY", "")
        if not key:
            raise ValueError(
                "MISTRAL_API_KEY no encontrada. "
                "Define la variable de entorno o pásala al constructor."
            )
        if model is None:
            model = os.environ.get("MISTRAL_MODEL", self.DEFAULT_MODEL)
        try:
            from mistralai.client.sdk import Mistral
            self.client = Mistral(api_key=key)
        except ImportError as exc:
            raise ImportError("pip install mistralai") from exc
        self.model = model
        logger.info(f"MistralFraudAnalyzer inicializado con modelo: {model}")

    def analyze(self, message: str, context: Optional[dict] = None) -> dict:
        user_content = f'Mensaje a analizar:\n"""\n{message}\n"""'

        if context:
            lines = []
            if "ml_label" in context:
                lines.append(
                    f"- Clasificador ML: {context['ml_label']} "
                    f"(confianza: {context.get('ml_confidence', 'N/A')})"
                )
            if "rule_score" in context:
                lines.append(f"- Puntaje sistema de reglas: {context['rule_score']}/100")
            if context.get("signals"):
                lines.append(f"- Señales detectadas: {', '.join(context['signals'])}")
            if lines:
                user_content += "\n\nContexto del sistema previo:\n" + "\n".join(lines)

        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                response_format={"type": "json_object"},
                max_tokens=400,
                temperature=0.05,
            )
            raw = response.choices[0].message.content
            result = json.loads(raw)
            result.setdefault("verdict",     "suspicious")
            result.setdefault("confidence",  0.5)
            result.setdefault("fraud_type",  "none")
            result.setdefault("indicators",  [])
            result.setdefault("explanation", "")
            logger.info(
                f"LLM resultado: verdict={result['verdict']}, "
                f"confidence={result['confidence']:.2f}, fraud_type={result['fraud_type']}"
            )
            return result

        except json.JSONDecodeError as exc:
            logger.error(f"Error parseando JSON del LLM: {exc}")
            return self._fallback("json_parse_error")
        except Exception as exc:
            logger.error(f"Error en Mistral API: {exc}")
            return self._fallback("api_error")

    @staticmethod
    def _fallback(error_type: str) -> dict:
        return {
            "verdict":     "suspicious",
            "confidence":  0.5,
            "fraud_type":  "none",
            "indicators":  [],
            "explanation": (
                f"Análisis LLM no disponible ({error_type}). "
                "Se mantiene el resultado del clasificador ML."
            ),
            "error": error_type,
        }
