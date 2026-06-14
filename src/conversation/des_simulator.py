"""
Simulador DES (Discrete Event Simulation) de conversaciones fraudulentas.

Fuente teórica: Simulación de Eventos Discretos con Calendario de Eventos (event-driven simulation).

Por qué DES mejora el entrenamiento del BiLSTM:
  - Las secuencias actuales en neural.py se ensamblan ALEATORIAMENTE (mensajes sueltos del pool).
  - DES genera conversaciones con ESTRUCTURA CAUSAL: el estafador siempre empieza con
    contacto → construye confianza → anuncia el problema → inyecta urgencia → pide credenciales.
  - Esto hace que los risk_features del encoder sigan un patrón de ESCALADA real (bajo → alto),
    exactamente el patrón que el BiLSTM necesita aprender.
  - Contexto compartido: mismo banco, misma credencial, mismo monto en todos los mensajes.

Arquitectura del simulador:
  - FraudEventType (Enum)    — 8 tipos de evento con rol en el fraude
  - FraudEvent (dataclass)   — evento con timestamp para heapq
  - ConversationStateMachine — matriz de transición con probabilidades y delays
  - DESConversationSimulator — motor principal: calendar + text generation + dataset output

Uso:
    sim = DESConversationSimulator(random_state=42)

    # Simular una conversación de fraude bancario
    events = sim.simulate_conversation(fraud_type="phishing_bancario", max_events=6)
    messages = sim.events_to_messages(events, fraud_type="phishing_bancario")

    # Generar dataset completo para reentrenar el BiLSTM
    df = sim.generate_dataset(n_conversations=500)
    df.to_csv("data/processed/des_conversations.csv", index=False)
"""

import heapq
import logging
import random
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from src.config import LABEL_COLUMN, MODELS_DIR, TEXT_COLUMN
from src.conversation.event_templates import (
    CONTEXT_POOLS,
    LEGIT_TEMPLATES,
    TEMPLATES,
)

logger = logging.getLogger(__name__)

DES_SIMULATOR_FILE = "des_simulator.joblib"


# ---------------------------------------------------------------------------
# Tipos de evento
# ---------------------------------------------------------------------------

class FraudEventType(Enum):
    """8 tipos de evento que componen el arco narrativo de un fraude."""
    CONTACT_INIT       = "contact_init"       # primer contacto inocente
    TRUST_BUILD        = "trust_build"        # genera confianza, suplanta identidad
    PROBLEM_ANNOUNCE   = "problem_announce"   # anuncia el "problema" central
    URGENCY_INJECT     = "urgency_inject"     # inyecta presión temporal
    CREDENTIAL_REQUEST = "credential_request" # solicita datos sensibles
    PAYMENT_PRESSURE   = "payment_pressure"   # exige transferencia / depósito
    THREAT_ESCALATE    = "threat_escalate"    # amenaza con consecuencias graves
    RECOVERY_OFFER     = "recovery_offer"     # ofrece la "solución" (cierre)


# ---------------------------------------------------------------------------
# Evento del calendario
# ---------------------------------------------------------------------------

@dataclass
class FraudEvent:
    """
    Un evento en el calendario DES.

    timestamp (float) — minutos desde el inicio de la conversación.
    context   (dict)  — variables compartidas: banco, credencial, monto, etc.
    __lt__ permite ordenar en heapq por timestamp.
    """
    event_type: FraudEventType
    timestamp:  float
    sender:     str       # "scammer" | "victim"
    context:    dict = field(default_factory=dict)

    def __lt__(self, other: "FraudEvent") -> bool:
        return self.timestamp < other.timestamp

    def __le__(self, other: "FraudEvent") -> bool:
        return self.timestamp <= other.timestamp


# ---------------------------------------------------------------------------
# Máquina de estados (tabla de transiciones)
# ---------------------------------------------------------------------------

class ConversationStateMachine:
    """
    Define las transiciones posibles entre tipos de evento.

    Cada entrada tiene la forma:
        event_type → [(next_type, probability, mean_delay_minutes), ...]

    Las probabilidades de cada estado suman 1.0 (salvo terminales = 0).
    El delay es la media de una distribución exponencial: random.expovariate(1/mean).
    Los eventos de alta urgencia (urgency_inject, credential_request) tienen
    factor de aceleración 0.6 (delays 40% más cortos que en contact/trust).
    """

    #: Transitions: {EventType → [(next_type_value, prob, mean_minutes), ...]}
    TRANSITIONS: dict[str, list[tuple[str, float, float]]] = {
        FraudEventType.CONTACT_INIT.value: [
            (FraudEventType.TRUST_BUILD.value,       0.60, 3.0),
            (FraudEventType.PROBLEM_ANNOUNCE.value,  0.40, 1.5),
        ],
        FraudEventType.TRUST_BUILD.value: [
            (FraudEventType.PROBLEM_ANNOUNCE.value,  0.50, 4.0),
            (FraudEventType.URGENCY_INJECT.value,    0.30, 2.0),
            (FraudEventType.RECOVERY_OFFER.value,    0.20, 3.0),
        ],
        FraudEventType.PROBLEM_ANNOUNCE.value: [
            (FraudEventType.URGENCY_INJECT.value,    0.50, 2.0),
            (FraudEventType.CREDENTIAL_REQUEST.value,0.30, 2.0),
            (FraudEventType.THREAT_ESCALATE.value,   0.20, 1.0),
        ],
        FraudEventType.URGENCY_INJECT.value: [
            (FraudEventType.CREDENTIAL_REQUEST.value,0.50, 1.5),
            (FraudEventType.PAYMENT_PRESSURE.value,  0.30, 1.5),
            (FraudEventType.THREAT_ESCALATE.value,   0.20, 1.0),
        ],
        FraudEventType.CREDENTIAL_REQUEST.value: [
            (FraudEventType.PAYMENT_PRESSURE.value,  0.50, 1.0),
            (FraudEventType.THREAT_ESCALATE.value,   0.30, 1.0),
            (FraudEventType.RECOVERY_OFFER.value,    0.20, 2.0),
        ],
        FraudEventType.PAYMENT_PRESSURE.value: [
            (FraudEventType.THREAT_ESCALATE.value,   0.60, 1.0),
            (FraudEventType.RECOVERY_OFFER.value,    0.40, 1.5),
        ],
        FraudEventType.THREAT_ESCALATE.value: [
            (FraudEventType.RECOVERY_OFFER.value,    1.00, 1.5),
        ],
        FraudEventType.RECOVERY_OFFER.value: [],  # terminal — no genera más eventos
    }

    def sample_next(
        self,
        current_type: FraudEventType,
        rng: random.Random,
    ) -> Optional[tuple[FraudEventType, float]]:
        """
        Muestrea el siguiente tipo de evento y el delay en minutos.

        Returns:
            (next_type, delay_minutes) o None si el estado es terminal.
        """
        transitions = self.TRANSITIONS.get(current_type.value, [])
        if not transitions:
            return None

        types  = [t[0] for t in transitions]
        probs  = [t[1] for t in transitions]
        means  = [t[2] for t in transitions]

        chosen_idx = rng.choices(range(len(types)), weights=probs, k=1)[0]
        next_val   = types[chosen_idx]
        mean_min   = means[chosen_idx]

        # Distribución exponencial: delay = expovariate(1/mean)
        delay = rng.expovariate(1.0 / max(mean_min, 0.1))

        return FraudEventType(next_val), delay


# ---------------------------------------------------------------------------
# Simulador DES
# ---------------------------------------------------------------------------

_SM = ConversationStateMachine()

# Tipos de fraude soportados (mismos que SpanishAugmenter para coherencia)
_FRAUD_TYPES = [
    "phishing_bancario",
    "sat_devolucion",
    "paqueteria_falsa",
    "premio_oxxo",
    "solicitud_otp",
    "empleo_falso",
    "compraventa_fraude",
]


class DESConversationSimulator:
    """
    Simula conversaciones de fraude usando un calendario de eventos DES.

    El motor garantiza:
    1. Orden causal: CONTACT_INIT siempre primero, escalada progresiva.
    2. Coherencia de contexto: mismo banco/credencial/monto en todos los mensajes.
    3. Timestamps realistas: distribución exponencial, aceleración en urgencia.
    4. Texto natural: templates con sustitución de variables + LLM opcional.

    Uso:
        sim = DESConversationSimulator(random_state=42)
        events = sim.simulate_conversation("phishing_bancario", max_events=6)
        msgs   = sim.events_to_messages(events, "phishing_bancario")
        df     = sim.generate_dataset(n_conversations=500)
    """

    def __init__(
        self,
        fraud_types:  Optional[list[str]] = None,
        use_llm:      bool = False,
        api_key:      Optional[str] = None,
        llm_model:    str  = "open-mistral-nemo",
        random_state: int  = 42,
    ) -> None:
        self.fraud_types  = fraud_types or _FRAUD_TYPES
        self.use_llm      = use_llm
        self._llm_model   = llm_model
        self._rng         = random.Random(random_state)
        self._random_state = random_state
        self._llm_client  = None

        if use_llm and api_key:
            try:
                from mistralai.client.sdk import Mistral
                self._llm_client = Mistral(api_key=api_key)
                logger.info(f"DES: LLM activo ({llm_model}).")
            except Exception as exc:
                logger.warning(f"DES: LLM no disponible: {exc}. Usando templates.")

    # ------------------------------------------------------------------
    # API principal
    # ------------------------------------------------------------------

    def simulate_conversation(
        self,
        fraud_type: Optional[str] = None,
        max_events: int = 6,
    ) -> list[FraudEvent]:
        """
        Simula una conversación de fraude usando el calendario de eventos DES.

        Garantías:
        - El primer evento siempre es CONTACT_INIT (t=0).
        - Los timestamps son estrictamente crecientes.
        - El contexto (banco, credencial, etc.) es compartido en todos los eventos.
        - Nunca supera max_events eventos.

        Args:
            fraud_type: Tipo de fraude. Si None, se elige aleatoriamente.
            max_events: Máximo de eventos en la conversación (default: 6).

        Returns:
            list[FraudEvent] ordenada por timestamp.
        """
        if fraud_type is None:
            fraud_type = self._rng.choice(self.fraud_types)

        context = self._sample_context(fraud_type)

        # Calendario: min-heap de FraudEvent ordenado por timestamp
        calendar: list[FraudEvent] = []
        heapq.heappush(calendar, FraudEvent(
            event_type=FraudEventType.CONTACT_INIT,
            timestamp=0.0,
            sender="scammer",
            context=context,
        ))

        events: list[FraudEvent] = []

        while calendar and len(events) < max_events:
            current = heapq.heappop(calendar)
            events.append(current)

            # Generar el siguiente evento según la máquina de estados
            result = _SM.sample_next(current.event_type, self._rng)
            if result is None:
                break  # estado terminal

            next_type, delay = result
            heapq.heappush(calendar, FraudEvent(
                event_type=next_type,
                timestamp=current.timestamp + delay,
                sender="scammer",
                context=context,
            ))

        return events

    def events_to_messages(
        self,
        events:     list[FraudEvent],
        fraud_type: str,
    ) -> list:
        """
        Convierte una lista de FraudEvent a objetos Message con texto natural.

        Args:
            events:     Lista de eventos (output de simulate_conversation).
            fraud_type: Tipo de fraude para seleccionar templates.

        Returns:
            list[Message] listos para el ConversationAnalyzer o el BiLSTM.
        """
        from src.conversation.models import Message

        messages = []
        for event in events:
            text = self._generate_text(event, fraud_type)
            ts   = f"{event.timestamp:.1f}min"
            messages.append(Message(text=text, sender=event.sender, timestamp=ts))
        return messages

    def generate_conversations(
        self,
        n:           int = 200,
        fraud_types: Optional[list[str]] = None,
    ) -> list[list]:
        """
        Genera n conversaciones de fraude completas como listas de Message.

        Útil para pasar directamente al ConversationAnalyzer o al ACO.

        Returns:
            list[list[Message]] — cada elemento es una conversación completa.
        """
        types = fraud_types or self.fraud_types
        conversations = []
        for i in range(n):
            ft = types[i % len(types)]
            events = self.simulate_conversation(fraud_type=ft)
            msgs   = self.events_to_messages(events, fraud_type=ft)
            conversations.append(msgs)
        logger.info(f"DES: generadas {n} conversaciones fraudulentas.")
        return conversations

    def generate_dataset(
        self,
        n_conversations: int   = 500,
        legit_ratio:     float = 0.5,
    ) -> pd.DataFrame:
        """
        Genera un dataset de conversaciones listo para sequence_model.fit().

        Columnas del DataFrame resultante:
          message          — texto del mensaje
          label            — "fraudulent" | "legitimate"
          source           — "des_fraud" | "des_legit"
          conversation_id  — identificador de la conversación
          turn_idx         — índice del turno dentro de la conversación

        Args:
            n_conversations: Total de conversaciones (fraude + legítimas).
            legit_ratio:     Fracción de conversaciones legítimas (default: 0.5).

        Returns:
            pd.DataFrame compatible con sequence_model.fit() (columnas message, label).
        """
        n_fraud = int(n_conversations * (1 - legit_ratio))
        n_legit = n_conversations - n_fraud

        rows: list[dict] = []
        conv_id = 0

        # --- Conversaciones de fraude ---
        for i in range(n_fraud):
            ft     = self.fraud_types[i % len(self.fraud_types)]
            events = self.simulate_conversation(fraud_type=ft)
            for turn, event in enumerate(events):
                text = self._generate_text(event, ft)
                rows.append({
                    TEXT_COLUMN:        text,
                    LABEL_COLUMN:       "fraudulent",
                    "source":           "des_fraud",
                    "conversation_id":  conv_id,
                    "turn_idx":         turn,
                    "event_type":       event.event_type.value,
                    "fraud_type":       ft,
                })
            conv_id += 1

        # --- Conversaciones legítimas ---
        for _ in range(n_legit):
            msgs = self._generate_legit_conversation()
            for turn, (text, category) in enumerate(msgs):
                rows.append({
                    TEXT_COLUMN:        text,
                    LABEL_COLUMN:       "legitimate",
                    "source":           "des_legit",
                    "conversation_id":  conv_id,
                    "turn_idx":         turn,
                    "event_type":       "legit",
                    "fraud_type":       category,
                })
            conv_id += 1

        df = pd.DataFrame(rows)
        n_msgs = len(df)
        n_fraud_msgs = (df[LABEL_COLUMN] == "fraudulent").sum()
        logger.info(
            f"DES dataset: {conv_id} conversaciones, {n_msgs} mensajes "
            f"({n_fraud_msgs} fraude / {n_msgs - n_fraud_msgs} legítimo)"
        )
        return df

    # ------------------------------------------------------------------
    # Generación de texto
    # ------------------------------------------------------------------

    def _generate_text(self, event: FraudEvent, fraud_type: str) -> str:
        """
        Genera el texto de un evento mediante templates (o LLM si está activo).
        Rellena los placeholders con el contexto compartido del evento.
        """
        if self.use_llm and self._llm_client:
            text = self._generate_text_llm(event, fraud_type)
            if text:
                return text

        return self._generate_text_template(event, fraud_type)

    def _generate_text_template(self, event: FraudEvent, fraud_type: str) -> str:
        """Genera texto rellenando un template con el contexto del evento."""
        event_key = event.event_type.value

        # Buscar templates: específico del tipo de fraude → fallback genérico
        bucket = TEMPLATES.get(event_key, {})
        variants = bucket.get(fraud_type) or bucket.get("_default", [])

        if not variants:
            return f"[Evento {event_key} para {fraud_type}]"

        template = self._rng.choice(variants)
        return self._fill_template(template, event.context)

    def _fill_template(self, template: str, context: dict) -> str:
        """Sustituye {placeholder} con valores del contexto."""
        result = template
        for key, value in context.items():
            result = result.replace(f"{{{key}}}", str(value))
        # Limpiar placeholders sin resolver (si algún tipo de fraude no tiene la variable)
        result = re.sub(r"\{[a-z_]+\}", "", result)
        return result.strip()

    def _generate_text_llm(self, event: FraudEvent, fraud_type: str) -> Optional[str]:
        """Genera texto usando Mistral. Devuelve None si falla."""
        prompt = (
            f"Genera UN mensaje de texto natural en español mexicano "
            f"para este evento de fraude:\n"
            f"  Tipo de fraude: {fraud_type}\n"
            f"  Tipo de evento: {event.event_type.value}\n"
            f"  Contexto: {event.context}\n\n"
            f"El mensaje debe sonar auténtico y tener máximo 60 palabras. "
            f"Solo responde con el mensaje, sin explicaciones."
        )
        try:
            resp = self._llm_client.chat.complete(
                model=self._llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            logger.debug(f"DES LLM error: {exc}")
            return None

    # ------------------------------------------------------------------
    # Contexto compartido
    # ------------------------------------------------------------------

    def _sample_context(self, fraud_type: str) -> dict:
        """
        Genera un diccionario de contexto consistente para toda la conversación.
        Todos los eventos de una misma sesión comparten el mismo banco, credencial, etc.
        """
        pool = CONTEXT_POOLS.get(fraud_type, {})
        return {
            key: self._rng.choice(values)
            for key, values in pool.items()
        }

    # ------------------------------------------------------------------
    # Conversaciones legítimas
    # ------------------------------------------------------------------

    def _generate_legit_conversation(
        self,
        min_turns: int = 2,
        max_turns: int = 5,
    ) -> list[tuple[str, str]]:
        """
        Genera una conversación legítima aleatoria.

        Returns:
            list[(text, category)] — mensajes con su categoría temática.
        """
        category  = self._rng.choice(list(LEGIT_TEMPLATES.keys()))
        pool      = LEGIT_TEMPLATES[category]
        n_turns   = self._rng.randint(min_turns, min(max_turns, len(pool)))
        chosen    = self._rng.sample(pool, n_turns)
        return [(text, category) for text in chosen]

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> Path:
        """Guarda el simulador (incluyendo estado del RNG) en disco."""
        out = Path(path) if path else MODELS_DIR / DES_SIMULATOR_FILE
        state = {
            "fraud_types":   self.fraud_types,
            "use_llm":       self.use_llm,
            "llm_model":     self._llm_model,
            "random_state":  self._random_state,
            "rng_state":     self._rng.getstate(),
        }
        joblib.dump(state, out)
        logger.info(f"DESConversationSimulator guardado en {out}")
        return out

    def load(self, path: Optional[Path] = None) -> None:
        """Restaura el simulador desde disco."""
        p = Path(path) if path else MODELS_DIR / DES_SIMULATOR_FILE
        if not p.exists():
            raise FileNotFoundError(f"DESConversationSimulator no encontrado: {p}")
        state = joblib.load(p)
        self.fraud_types   = state["fraud_types"]
        self.use_llm       = state["use_llm"]
        self._llm_model    = state["llm_model"]
        self._random_state = state["random_state"]
        self._rng.setstate(state["rng_state"])
        logger.info(f"DESConversationSimulator cargado desde {p}")
