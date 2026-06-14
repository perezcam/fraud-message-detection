"""Tests para DESConversationSimulator (Fase 5 - Módulo DES)."""

import heapq
import pytest

from src.conversation.des_simulator import (
    DESConversationSimulator,
    FraudEvent,
    FraudEventType,
    ConversationStateMachine,
    _FRAUD_TYPES,
)
from src.config import TEXT_COLUMN, LABEL_COLUMN


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sim():
    return DESConversationSimulator(random_state=0)


# ---------------------------------------------------------------------------
# Tests de FraudEvent y heapq
# ---------------------------------------------------------------------------

def test_fraud_event_ordering_in_heapq():
    """FraudEvent con __lt__ se ordena correctamente en heapq."""
    ctx = {}
    e1 = FraudEvent(FraudEventType.URGENCY_INJECT, 5.0, "scammer", ctx)
    e2 = FraudEvent(FraudEventType.CONTACT_INIT,   0.0, "scammer", ctx)
    e3 = FraudEvent(FraudEventType.THREAT_ESCALATE,3.0, "scammer", ctx)

    heap = [e1, e2, e3]
    heapq.heapify(heap)

    popped = [heapq.heappop(heap) for _ in range(3)]
    timestamps = [e.timestamp for e in popped]
    assert timestamps == sorted(timestamps), "heapq no devuelve eventos en orden de timestamp"


def test_fraud_event_lt():
    ctx = {}
    early = FraudEvent(FraudEventType.CONTACT_INIT, 0.0, "scammer", ctx)
    late  = FraudEvent(FraudEventType.RECOVERY_OFFER, 10.0, "scammer", ctx)
    assert early < late
    assert not late < early


# ---------------------------------------------------------------------------
# Tests de simulate_conversation()
# ---------------------------------------------------------------------------

def test_simulate_starts_with_contact_init(sim):
    """El primer evento de cualquier conversación siempre es CONTACT_INIT."""
    for ft in _FRAUD_TYPES:
        events = sim.simulate_conversation(fraud_type=ft, max_events=4)
        assert len(events) >= 1
        assert events[0].event_type == FraudEventType.CONTACT_INIT, (
            f"Fraude {ft}: primer evento es {events[0].event_type}, esperaba CONTACT_INIT"
        )


def test_simulate_min_events(sim):
    """Se generan al menos 2 eventos por conversación."""
    events = sim.simulate_conversation(max_events=6)
    assert len(events) >= 2


def test_simulate_max_events(sim):
    """Nunca se superan max_events eventos."""
    for max_ev in [2, 4, 8]:
        events = sim.simulate_conversation(max_events=max_ev)
        assert len(events) <= max_ev, f"max_events={max_ev} pero got {len(events)}"


def test_timestamps_monotonic(sim):
    """Los timestamps de los eventos son estrictamente crecientes."""
    for _ in range(10):
        events = sim.simulate_conversation(max_events=6)
        timestamps = [e.timestamp for e in events]
        assert timestamps == sorted(timestamps), f"Timestamps no monotónicos: {timestamps}"


def test_context_shared_across_events(sim):
    """Todos los eventos de una conversación comparten el mismo contexto (mismo dict id)."""
    events = sim.simulate_conversation(fraud_type="phishing_bancario", max_events=5)
    if len(events) < 2:
        pytest.skip("Conversación muy corta para verificar contexto compartido")
    # El contexto debe ser el mismo objeto (mismo banco en todos los mensajes)
    ctx0 = events[0].context
    for e in events[1:]:
        assert e.context is ctx0, "Contexto no compartido entre eventos"


def test_random_fraud_type_works(sim):
    """simulate_conversation sin fraud_type elige uno aleatoriamente sin error."""
    for _ in range(5):
        events = sim.simulate_conversation(fraud_type=None, max_events=4)
        assert len(events) >= 1


# ---------------------------------------------------------------------------
# Tests de events_to_messages()
# ---------------------------------------------------------------------------

def test_events_to_messages_count(sim):
    """events_to_messages devuelve tantos mensajes como eventos."""
    events = sim.simulate_conversation("phishing_bancario", max_events=5)
    msgs   = sim.events_to_messages(events, "phishing_bancario")
    assert len(msgs) == len(events)


def test_events_to_messages_non_empty(sim):
    """Ningún mensaje tiene texto vacío."""
    for ft in _FRAUD_TYPES:
        events = sim.simulate_conversation(ft, max_events=4)
        msgs   = sim.events_to_messages(events, ft)
        for m in msgs:
            assert m.text.strip(), f"Texto vacío en fraude {ft}"


def test_events_to_messages_are_message_objects(sim):
    """events_to_messages devuelve objetos Message."""
    from src.conversation.models import Message
    events = sim.simulate_conversation("sat_devolucion", max_events=3)
    msgs   = sim.events_to_messages(events, "sat_devolucion")
    for m in msgs:
        assert isinstance(m, Message)


# ---------------------------------------------------------------------------
# Tests de generate_dataset()
# ---------------------------------------------------------------------------

def test_generate_dataset_columns(sim):
    """DataFrame tiene al menos las columnas requeridas por sequence_model.fit()."""
    df = sim.generate_dataset(n_conversations=10)
    assert TEXT_COLUMN  in df.columns
    assert LABEL_COLUMN in df.columns


def test_generate_dataset_fraud_label(sim):
    """Mensajes de conversaciones de fraude tienen label 'fraudulent'."""
    df = sim.generate_dataset(n_conversations=10, legit_ratio=0.0)
    assert (df[LABEL_COLUMN] == "fraudulent").all()


def test_generate_dataset_legit_label(sim):
    """Mensajes de conversaciones legítimas tienen label 'legitimate'."""
    df = sim.generate_dataset(n_conversations=10, legit_ratio=1.0)
    assert (df[LABEL_COLUMN] == "legitimate").all()


def test_generate_dataset_has_both_labels(sim):
    """Con legit_ratio=0.5 el dataset tiene ambas etiquetas."""
    df = sim.generate_dataset(n_conversations=20, legit_ratio=0.5)
    labels = df[LABEL_COLUMN].unique().tolist()
    assert "fraudulent" in labels
    assert "legitimate" in labels


def test_all_fraud_types_work(sim):
    """Los 7 tipos de fraude generan conversaciones y mensajes sin error."""
    for ft in _FRAUD_TYPES:
        events = sim.simulate_conversation(fraud_type=ft, max_events=5)
        msgs   = sim.events_to_messages(events, ft)
        assert len(msgs) >= 1, f"Sin mensajes para {ft}"


def test_reproducibility():
    """El mismo random_state produce exactamente las mismas conversaciones."""
    sim1 = DESConversationSimulator(random_state=7)
    sim2 = DESConversationSimulator(random_state=7)

    ev1 = sim1.simulate_conversation("phishing_bancario", max_events=5)
    ev2 = sim2.simulate_conversation("phishing_bancario", max_events=5)

    assert len(ev1) == len(ev2)
    for e1, e2 in zip(ev1, ev2):
        assert e1.event_type == e2.event_type
        assert abs(e1.timestamp - e2.timestamp) < 1e-10


# ---------------------------------------------------------------------------
# Test de save/load
# ---------------------------------------------------------------------------

def test_save_and_load(tmp_path, sim):
    """save() y load() preservan el simulador."""
    out = tmp_path / "des_test.joblib"
    sim.save(path=out)
    assert out.exists()

    sim2 = DESConversationSimulator(random_state=99)
    sim2.load(path=out)
    assert sim2.fraud_types == sim.fraud_types

    # Mismos resultados tras restaurar el RNG
    ev2 = sim2.simulate_conversation("phishing_bancario", max_events=3)
    assert len(ev2) >= 1
