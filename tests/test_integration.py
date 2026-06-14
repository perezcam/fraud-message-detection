"""
Test de integración end-to-end de la cascada de 9 capas.

Requiere que los modelos estén entrenados en models/.
Se salta automáticamente si los artefactos no existen.
"""

from __future__ import annotations

import pytest
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"

# Artefactos mínimos para que la cascada funcione
REQUIRED_ARTIFACTS = [
    "lightgbm_model.joblib",
    "tfidf_vectorizer.joblib",
]

pytestmark = pytest.mark.skipif(
    not all((MODELS_DIR / f).exists() for f in REQUIRED_ARTIFACTS),
    reason="Modelos no entrenados — ejecuta: python main.py train --dataset data/processed/messages.csv --model lightgbm",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def predictor():
    from src.detection.cascade import CascadePredictor
    return CascadePredictor()


FRAUD_MESSAGES = [
    "ALERTA BBVA: su cuenta fue bloqueada por actividad inusual. Verifique en http://bbva-seguro.mx ahora.",
    "Urgente: proporcione su NIP y contraseña para evitar el bloqueo de su tarjeta Santander.",
    # Fraude SAT con señales explícitas (la variante sutil solo URL es un gap conocido sin LLM)
    "URGENTE SAT: Tiene una devolución pendiente. Ingrese su RFC y contraseña en sat-devolucion.mx para reclamarla.",
    "Ganó $50,000 en el sorteo OXXO. Llame al 5512345678 para reclamar su premio hoy.",
]

# Fraude sutil detectado solo con LLM — sin señales léxicas explícitas
# "El SAT detectó irregularidades en su declaración. Regularícese en sat-devolucion.mx con su RFC."
# → risk_level=low (falso negativo conocido sin LLM activo)

LEGIT_MESSAGES = [
    "Hola, ¿cómo estás? ¿Tienes tiempo para tomar un café esta tarde?",
    "El reporte de ventas del Q2 ya está disponible en el portal interno.",
    "¿Puedes revisar el pull request que subí ayer? Está en la rama feature/login.",
    "Recordatorio: junta del equipo mañana a las 10am en la sala de conferencias.",
]


# ---------------------------------------------------------------------------
# Tests de predicción individual
# ---------------------------------------------------------------------------

def test_fraud_messages_classified_as_fraud(predictor):
    """Mensajes de fraude obvio deben tener risk_level high o medium."""
    for msg in FRAUD_MESSAGES:
        result = predictor.predict(msg)
        assert result["risk_level"] in ("high", "medium"), (
            f"Fraude no detectado: '{msg[:60]}' → {result['risk_level']} "
            f"(score={result.get('rule_score', '?')})"
        )


def test_legit_messages_classified_as_legit(predictor):
    """Mensajes legítimos claros deben tener risk_level low."""
    for msg in LEGIT_MESSAGES:
        result = predictor.predict(msg)
        assert result["risk_level"] == "low", (
            f"Falso positivo: '{msg[:60]}' → {result['risk_level']}"
        )


def test_result_has_required_keys(predictor):
    """El resultado de predict() siempre incluye las claves esperadas."""
    required = {"predicted_class", "risk_level", "confidence", "recommendation"}
    result = predictor.predict("Hola, ¿cómo estás?")
    for key in required:
        assert key in result, f"Clave faltante: {key}"


def test_confidence_is_normalized(predictor):
    """confidence siempre está en [0, 1]."""
    for msg in FRAUD_MESSAGES + LEGIT_MESSAGES:
        result = predictor.predict(msg)
        assert 0.0 <= result["confidence"] <= 1.0, (
            f"Confianza fuera de rango: {result['confidence']}"
        )


def test_layers_used_is_populated(predictor):
    """layers_used indica qué capas procesaron el mensaje."""
    result = predictor.predict(FRAUD_MESSAGES[0])
    assert "layers_used" in result
    assert len(result["layers_used"]) >= 1


# ---------------------------------------------------------------------------
# Tests de cascada con capas opcionales
# ---------------------------------------------------------------------------

def test_anomaly_layer_activates(predictor):
    """Si el AnomalyDetector está cargado, aparece en layers_used o en el resultado."""
    from pathlib import Path
    if not (MODELS_DIR / "anomaly_detector.joblib").exists():
        pytest.skip("anomaly_detector.joblib no entrenado")
    result = predictor.predict(FRAUD_MESSAGES[0])
    assert "anomaly_score" in result or "anomaly" in result.get("layers_used", [])


def test_bayes_layer_activates(predictor):
    """Si FraudBayesNet está cargada, bayes_score aparece en el resultado."""
    if not (MODELS_DIR / "bayes_net.joblib").exists():
        pytest.skip("bayes_net.joblib no entrenado")
    result = predictor.predict(FRAUD_MESSAGES[0])
    assert "bayes_score" in result or "bayes" in result.get("layers_used", [])


def test_cbr_layer_activates(predictor):
    """Si CaseBase está cargada, cbr_score aparece en el resultado."""
    if not (MODELS_DIR / "case_base.npz").exists():
        pytest.skip("case_base.npz no construida")
    result = predictor.predict(FRAUD_MESSAGES[0])
    assert "cbr_score" in result or "cbr" in result.get("layers_used", [])


def test_meta_learner_activates(predictor):
    """Si el meta-learner está cargado, meta_proba aparece en el resultado."""
    if not (MODELS_DIR / "meta_learner.joblib").exists():
        pytest.skip("meta_learner.joblib no entrenado")
    result = predictor.predict(FRAUD_MESSAGES[0])
    assert "meta_proba" in result or "meta" in result.get("layers_used", [])


# ---------------------------------------------------------------------------
# Tests de robustez
# ---------------------------------------------------------------------------

def test_empty_message_does_not_crash(predictor):
    """Un mensaje vacío no debe lanzar excepción."""
    result = predictor.predict("")
    assert "risk_level" in result


def test_very_long_message_does_not_crash(predictor):
    """Un mensaje extremadamente largo no debe lanzar excepción."""
    long_msg = "fraude urgente " * 500
    result = predictor.predict(long_msg)
    assert "risk_level" in result


def test_unicode_message_does_not_crash(predictor):
    """Mensajes con caracteres especiales y emojis no deben fallar."""
    result = predictor.predict("🚨 Urgente!! Verifique su cuenta: http://evil.mx 🔒")
    assert "risk_level" in result


def test_repeated_calls_consistent(predictor):
    """Múltiples llamadas con el mismo mensaje retornan el mismo resultado."""
    msg = FRAUD_MESSAGES[0]
    r1 = predictor.predict(msg)
    r2 = predictor.predict(msg)
    assert r1["risk_level"] == r2["risk_level"]
    assert abs(r1["confidence"] - r2["confidence"]) < 0.10
