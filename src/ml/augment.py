"""
Augmentación de datos en español para el sistema de detección de fraude.

Estrategias implementadas:
  1. Generación sintética con Mistral LLM (few-shot prompting)
       — Genera mensajes de fraude y legítimos en español mexicano
       — Cubre 7 tipos de fraude comunes en LATAM
  2. Pseudo-labeling
       — Toma un corpus no etiquetado en español
       — Predice con el modelo actual
       — Mensajes con confianza > threshold → etiquetas automáticas
       — Retorna DataFrame listo para reentrenamiento
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Tipos de fraude y contexto para la generación
# ---------------------------------------------------------------------------

FRAUD_TYPES_ES = [
    {
        "type": "phishing_bancario",
        "description": "phishing bancario suplantando a BBVA, Santander, HSBC, Citibanamex o Banorte",
        "examples": [
            "ALERTA BBVA: Su cuenta fue bloqueada. Verifique sus datos en http://bbva-seguro.com",
            "Santander: detectamos un cargo no reconocido de $3,500. Confirme en https://bit.ly/sant-mx",
        ],
    },
    {
        "type": "sat_devolucion",
        "description": "suplantación del SAT ofreciendo una devolución de impuestos falsa",
        "examples": [
            "SAT MEXICO: Tiene un saldo a favor de $8,200. Ingrese su RFC en sat-devolucion.gob-mx.com",
        ],
    },
    {
        "type": "paqueteria_falsa",
        "description": "smishing de paquetería falsa (FedEx, DHL, Correos México, Estafeta)",
        "examples": [
            "FedEx: Su paquete #MX9284 está retenido. Pague $35 de arancel en fedex-liberacion.com",
        ],
    },
    {
        "type": "premio_oxxo",
        "description": "smishing de premio falso de OXXO, Walmart, Liverpool, Soriana o Coppel",
        "examples": [
            "¡Felicidades! Ganaste $10,000 en el sorteo OXXO. Reclama en oxxo-premios.com/ganador",
        ],
    },
    {
        "type": "solicitud_otp",
        "description": "ingeniería social para obtener código OTP, NIP o contraseña de forma urgente",
        "examples": [
            "Soy del banco, necesito que me pases el código que te llegó por SMS, es urgente o tu cuenta se bloquea",
        ],
    },
    {
        "type": "empleo_falso",
        "description": "oferta de empleo falsa que pide cuenta bancaria o datos personales por adelantado",
        "examples": [
            "Trabaja desde casa y gana $2,500 diarios. Solo necesitas enviar tu CLABE para el primer pago. WhatsApp: 555-1234",
        ],
    },
    {
        "type": "compraventa_fraude",
        "description": "fraude en compraventa por Facebook Marketplace o MercadoLibre solicitando datos bancarios",
        "examples": [
            "Hola, me interesa lo que vendes. Te pago por transferencia pero necesito tu número de cuenta y foto de tu INE primero",
        ],
    },
]

LEGIT_CONTEXTS_ES = [
    "Recordatorio de cita médica o dental sin solicitar datos",
    "Notificación de pedido en tienda física listo para recoger",
    "Mensaje de familiar o amigo sobre planes sociales",
    "Aviso de servicio (agua, luz, internet) sin link sospechoso",
    "Confirmación de reservación de restaurante o hotel",
    "Mensaje de trabajo sobre reunión o entrega de proyecto",
    "Saludo o felicitación personal sin solicitudes",
]

_SYSTEM_PROMPT_FRAUD = """\
Eres un generador de datos de entrenamiento para sistemas de detección de fraude.
Genera mensajes SMS o WhatsApp FRAUDULENTOS auténticos en español mexicano coloquial.
Los mensajes deben ser realistas, breves (1-3 oraciones), y representar fielmente \
el tipo de fraude especificado.
Responde ÚNICAMENTE con un objeto JSON: {"messages": ["msg1", "msg2", ...]}
No incluyas explicaciones ni texto adicional fuera del JSON.
"""

_SYSTEM_PROMPT_LEGIT = """\
Eres un generador de datos de entrenamiento para sistemas de detección de fraude.
Genera mensajes SMS o WhatsApp LEGÍTIMOS y cotidianos en español mexicano coloquial.
Los mensajes deben ser completamente inofensivos, sin solicitudes de datos, sin URLs \
sospechosas ni urgencia artificial.
Responde ÚNICAMENTE con un objeto JSON: {"messages": ["msg1", "msg2", ...]}
No incluyas explicaciones ni texto adicional fuera del JSON.
"""


class SpanishAugmenter:
    """
    Genera mensajes sintéticos en español usando Mistral AI (few-shot prompting)
    y realiza pseudo-labeling sobre corpora no etiquetados.

    Requiere MISTRAL_API_KEY configurada en .env

    Uso:
        augmenter = SpanishAugmenter()
        df = augmenter.augment_dataset(n_fraud=300, n_legit=200)
        df.to_csv("data/processed/messages_es_augmented.csv", index=False)
    """

    DEFAULT_MODEL = "open-mistral-nemo"
    BATCH_SIZE    = 10   # mensajes por llamada a la API

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        key = api_key or os.environ.get("MISTRAL_API_KEY", "")
        if not key:
            raise ValueError(
                "MISTRAL_API_KEY no encontrada. "
                "Configura la variable de entorno o pásala al constructor."
            )
        if model is None:
            model = os.environ.get("MISTRAL_MODEL", self.DEFAULT_MODEL)

        try:
            from mistralai.client.sdk import Mistral
            self._client = Mistral(api_key=key)
        except ImportError as exc:
            raise ImportError("pip install mistralai") from exc

        self._model = model
        logger.info(f"SpanishAugmenter inicializado con modelo: {model}")

    # ------------------------------------------------------------------
    # Generación de mensajes fraudulentos
    # ------------------------------------------------------------------

    def generate_fraud_messages(self, n_per_type: int = 10) -> list[str]:
        """
        Genera n_per_type mensajes de fraude por cada tipo definido en FRAUD_TYPES_ES.
        Retorna lista de strings.
        """
        all_messages = []
        for fraud_type in FRAUD_TYPES_ES:
            examples_str = "\n".join(f'  - "{ex}"' for ex in fraud_type["examples"])
            user_prompt = (
                f"Tipo de fraude: {fraud_type['description']}\n\n"
                f"Ejemplos de referencia:\n{examples_str}\n\n"
                f"Genera {n_per_type} mensajes DIFERENTES y realistas de este tipo. "
                f"Varía el tono, la urgencia y los detalles específicos. "
                f"Responde con el JSON: {{\"messages\": [\"msg1\", ...]}}"
            )
            msgs = self._generate_batch(_SYSTEM_PROMPT_FRAUD, user_prompt, n_per_type)
            all_messages.extend(msgs)
            logger.info(
                f"  {fraud_type['type']}: {len(msgs)} mensajes generados "
                f"(total acumulado: {len(all_messages)})"
            )
        return all_messages

    # ------------------------------------------------------------------
    # Generación de mensajes legítimos
    # ------------------------------------------------------------------

    def generate_legitimate_messages(self, n: int = 100) -> list[str]:
        """
        Genera n mensajes legítimos variados.
        """
        n_per_ctx = max(1, n // len(LEGIT_CONTEXTS_ES))
        all_messages = []
        for ctx in LEGIT_CONTEXTS_ES:
            user_prompt = (
                f"Contexto: {ctx}\n\n"
                f"Genera {n_per_ctx} mensajes cotidianos y completamente inofensivos. "
                f"Responde con el JSON: {{\"messages\": [\"msg1\", ...]}}"
            )
            msgs = self._generate_batch(_SYSTEM_PROMPT_LEGIT, user_prompt, n_per_ctx)
            all_messages.extend(msgs)
        return all_messages[:n]

    # ------------------------------------------------------------------
    # Pseudo-labeling
    # ------------------------------------------------------------------

    def pseudo_label(
        self,
        corpus_path: Path,
        predictor,
        threshold: float = 0.90,
        text_col: str = "message",
    ) -> pd.DataFrame:
        """
        Toma un CSV/TXT de mensajes no etiquetados en español y asigna pseudo-etiquetas
        a los que tengan confianza >= threshold.

        Args:
            corpus_path: Ruta al CSV (con columna text_col) o TXT (un mensaje por línea).
            predictor:   Instancia de FraudPredictor ya cargada.
            threshold:   Umbral de confianza mínimo para aceptar la pseudo-etiqueta.
            text_col:    Nombre de la columna de texto en el CSV.

        Returns:
            DataFrame con columnas "message" y "label" para los mensajes aceptados.
        """
        corpus_path = Path(corpus_path)
        if corpus_path.suffix == ".txt":
            texts = [line.strip() for line in corpus_path.read_text("utf-8").splitlines()
                     if line.strip()]
        else:
            df_raw = pd.read_csv(corpus_path)
            texts = df_raw[text_col].astype(str).tolist()

        accepted = []
        for text in texts:
            try:
                result = predictor.predict(text)
                conf   = result.get("confidence") or 0.0
                label  = result.get("predicted_class", "")
                if conf >= threshold and label in ("fraudulent", "legitimate"):
                    accepted.append({"message": text, "label": label})
            except Exception:
                continue

        logger.info(
            f"Pseudo-labeling: {len(accepted)}/{len(texts)} mensajes aceptados "
            f"(umbral={threshold})"
        )
        return pd.DataFrame(accepted)

    # ------------------------------------------------------------------
    # Pipeline completo
    # ------------------------------------------------------------------

    def augment_dataset(
        self,
        n_fraud: int = 300,
        n_legit: int = 200,
        base_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Genera un dataset augmentado con mensajes sintéticos en español.

        Args:
            n_fraud:  Total de mensajes fraudulentos a generar.
            n_legit:  Total de mensajes legítimos a generar.
            base_df:  DataFrame base opcional para concatenar (ej. dataset original en inglés).

        Returns:
            DataFrame con columnas "message", "label", "source".
        """
        n_per_type = max(1, n_fraud // len(FRAUD_TYPES_ES))
        logger.info(
            f"Generando ~{n_per_type} mensajes por tipo de fraude "
            f"({len(FRAUD_TYPES_ES)} tipos) y {n_legit} legítimos..."
        )

        fraud_msgs = self.generate_fraud_messages(n_per_type=n_per_type)[:n_fraud]
        legit_msgs = self.generate_legitimate_messages(n=n_legit)

        df_fraud = pd.DataFrame({
            "message": fraud_msgs,
            "label":   "fraudulent",
            "source":  "synthetic_es",
        })
        df_legit = pd.DataFrame({
            "message": legit_msgs,
            "label":   "legitimate",
            "source":  "synthetic_es",
        })

        frames = [df_fraud, df_legit]
        if base_df is not None:
            base_df = base_df.copy()
            if "source" not in base_df.columns:
                base_df["source"] = "original"
            frames.append(base_df)

        result = pd.concat(frames, ignore_index=True).sample(
            frac=1, random_state=42
        )
        dist = result["label"].value_counts().to_dict()
        logger.info(f"Dataset augmentado: {len(result)} mensajes — {dist}")
        return result

    # ------------------------------------------------------------------
    # Llamada a la API
    # ------------------------------------------------------------------

    def _generate_batch(
        self,
        system_prompt: str,
        user_prompt: str,
        expected_n: int,
    ) -> list[str]:
        """Llama a Mistral y extrae la lista de mensajes del JSON."""
        try:
            resp = self._client.chat.complete(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=1500,
                temperature=0.8,
            )
            raw = resp.choices[0].message.content
            data = json.loads(raw)
            msgs = data.get("messages", [])
            if not isinstance(msgs, list):
                return []
            return [str(m).strip() for m in msgs if m and str(m).strip()]
        except Exception as exc:
            logger.error(f"Error generando batch: {exc}")
            return []
