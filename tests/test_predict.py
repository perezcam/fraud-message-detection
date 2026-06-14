"""
Tests para el módulo de análisis de riesgo y estructura de predicción.
No requieren modelo entrenado: se prueban las capas de reglas y utilidades.
"""

import pytest

from src.rules.risk import analyze_risk


class TestAnalyzeRisk:
    def test_empty_string_returns_zero(self):
        result = analyze_risk("")
        assert result == {"risk_score": 0, "risk_level": "low", "signals": []}

    def test_returns_required_keys(self):
        result = analyze_risk("test message")
        assert "risk_score" in result
        assert "risk_level" in result
        assert "signals" in result

    def test_risk_level_valid_values(self):
        result = analyze_risk("cualquier texto")
        assert result["risk_level"] in ("low", "medium", "high")

    def test_risk_score_bounded(self):
        msg = (
            "URGENTE bloqueado transferencia contraseña PIN OTP "
            "http://evil.com $50,000 felicidades ganador!!!"
        )
        result = analyze_risk(msg)
        assert 0 <= result["risk_score"] <= 100

    def test_url_signal_detected(self):
        result = analyze_risk("Click aquí: https://evil.com/login")
        signals_text = " ".join(result["signals"]).lower()
        assert "url" in signals_text
        assert result["risk_score"] > 0

    def test_credential_signal_detected(self):
        result = analyze_risk("Envíenos su contraseña y PIN para verificar su cuenta")
        signals_text = " ".join(result["signals"]).lower()
        assert any(kw in signals_text for kw in ("contraseña", "pin", "código", "credencial", "solicita"))
        assert result["risk_score"] >= 20

    def test_urgency_signal_detected(self):
        result = analyze_risk("Actúe urgente, su cuenta será suspendida inmediatamente")
        assert result["risk_score"] > 0
        assert len(result["signals"]) > 0

    def test_legitimate_message_low_risk(self):
        result = analyze_risk("Hola, ya está listo tu pedido. Pasa a recogerlo.")
        assert result["risk_level"] in ("low", "medium")

    def test_high_risk_fraud_message(self):
        message = (
            "URGENTE: Su cuenta será BLOQUEADA. Verifique sus datos en "
            "http://banco-falso.com. Envíe contraseña, PIN y OTP ahora. "
            "¡Recibirá $10,000 de regalo!!!"
        )
        result = analyze_risk(message)
        assert result["risk_level"] == "high"
        assert result["risk_score"] >= 60
        assert len(result["signals"]) >= 3

    def test_signals_is_list(self):
        result = analyze_risk("test")
        assert isinstance(result["signals"], list)

    def test_risk_score_is_int(self):
        result = analyze_risk("Su cuenta está bloqueada, verifique aquí http://x.com")
        assert isinstance(result["risk_score"], int)

    def test_phone_signal_detected(self):
        result = analyze_risk("Llame al 5512345678 para reclamar su premio")
        signals_text = " ".join(result["signals"]).lower()
        assert "teléfono" in signals_text or result["risk_score"] > 0

    def test_money_signal_detected(self):
        result = analyze_risk("Reciba $5,000 en su cuenta ahora")
        signals_text = " ".join(result["signals"]).lower()
        assert "monetari" in signals_text or "financier" in signals_text or result["risk_score"] > 0

    def test_prize_signal_detected(self):
        result = analyze_risk("Felicidades, usted es el ganador del sorteo especial")
        assert result["risk_score"] > 0

    def test_transfer_signal_detected(self):
        result = analyze_risk("Realice la transferencia a nuestra cuenta bancaria de inmediato")
        assert result["risk_score"] > 0
