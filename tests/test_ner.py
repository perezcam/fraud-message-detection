"""Tests para src/ml/ner.py"""

import pytest
from src.ml.ner import extract_entities


class TestExtractEntities:
    def test_returns_required_keys(self):
        result = extract_entities("Hola mundo")
        for key in ("phones", "urls", "amounts", "accounts", "banks",
                    "entity_count", "has_phone", "has_url",
                    "has_amount", "has_bank", "has_account"):
            assert key in result

    def test_phone_detection(self):
        result = extract_entities("Llame al +52 55 1234 5678 para verificar")
        assert result["has_phone"] == 1.0
        assert len(result["phones"]) >= 1

    def test_url_detection(self):
        result = extract_entities("Haga clic en https://bbva-seguro.com/login")
        assert result["has_url"] == 1.0
        assert any("bbva" in u for u in result["urls"])

    def test_amount_detection(self):
        result = extract_entities("Deposite $5,000 pesos ahora")
        assert result["has_amount"] == 1.0
        assert len(result["amounts"]) >= 1

    def test_bank_detection(self):
        result = extract_entities("Su cuenta de BBVA fue bloqueada")
        assert result["has_bank"] == 1.0
        assert "bbva" in [b.lower() for b in result["banks"]]

    def test_no_entities_in_clean_text(self):
        result = extract_entities("Buenos días, cómo está usted")
        assert result["entity_count"] == 0
        assert result["has_phone"] == 0.0
        assert result["has_url"] == 0.0
        assert result["has_amount"] == 0.0

    def test_date_not_counted_as_account(self):
        result = extract_entities("El evento es en 2025")
        assert result["has_account"] == 0.0

    def test_multiple_entities(self):
        text = "Llame al 5512345678 o visite http://fraude.com. Santander bloqueó su cuenta"
        result = extract_entities(text)
        assert result["entity_count"] >= 3
        assert result["has_phone"] == 1.0
        assert result["has_url"] == 1.0
        assert result["has_bank"] == 1.0

    def test_empty_string(self):
        result = extract_entities("")
        assert result["entity_count"] == 0

    def test_short_phone_not_matched(self):
        result = extract_entities("código 1234")
        assert result["has_phone"] == 0.0
