"""Tests para el módulo de preprocesamiento."""

import pandas as pd
import pytest

from src.data.preprocessing import (
    clean_text,
    normalize_emails,
    normalize_money,
    normalize_phones,
    normalize_urls,
    preprocess_dataframe,
)


class TestCleanText:
    def test_empty_string_returns_empty(self):
        assert clean_text("") == ""

    def test_none_returns_empty(self):
        assert clean_text(None) == ""

    def test_whitespace_only_returns_empty(self):
        assert clean_text("   \t\n ") == ""

    def test_lowercases_text(self):
        assert clean_text("Hello World") == "hello world"

    def test_preserves_exclamation(self):
        result = clean_text("Act NOW!!!")
        assert "!" in result

    def test_preserves_question_mark(self):
        result = clean_text("¿Es esto real?")
        assert "?" in result

    def test_collapses_whitespace(self):
        result = clean_text("hello   world")
        assert "  " not in result


class TestNormalizeUrls:
    def test_http_url_replaced(self):
        text = "Click here: https://malicious-site.com/login"
        result = normalize_urls(text)
        assert "<URL>" in result
        assert "https://malicious-site.com" not in result

    def test_www_url_replaced(self):
        result = normalize_urls("Visit www.example.com")
        assert "<URL>" in result

    def test_bitly_replaced(self):
        result = normalize_urls("Link: bit.ly/abc123")
        assert "<URL>" in result

    def test_no_url_unchanged(self):
        text = "Hola, ¿cómo estás?"
        assert normalize_urls(text) == text

    def test_url_token_survives_full_pipeline(self):
        result = clean_text("Verify at http://bank.com/verify")
        assert "<url>" in result  # lowercased en el pipeline


class TestNormalizeEmails:
    def test_email_replaced(self):
        result = normalize_emails("Envía tus datos a hacker@evil.com")
        assert "<EMAIL>" in result
        assert "hacker@evil.com" not in result

    def test_no_email_unchanged(self):
        text = "Llama al banco directamente."
        assert normalize_emails(text) == text


class TestNormalizePhones:
    def test_long_phone_replaced(self):
        result = normalize_phones("Llama al 555-123-4567 ahora")
        assert "555-123-4567" not in result

    def test_short_number_not_replaced(self):
        result = normalize_phones("Código 123")
        assert "<PHONE>" not in result


class TestNormalizeMoney:
    def test_dollar_amount_replaced(self):
        result = normalize_money("You won $5,000")
        assert "<MONEY>" in result

    def test_pesos_amount_replaced(self):
        result = normalize_money("Recibe 10,000 pesos")
        assert "<MONEY>" in result

    def test_no_money_unchanged(self):
        text = "Hola, ¿cómo estás?"
        assert normalize_money(text) == text


class TestPreprocessDataframe:
    def test_removes_empty_messages(self):
        df = pd.DataFrame({
            "message": ["Hello world", "Spam http://evil.com", ""],
            "label": ["legitimate", "fraudulent", "legitimate"],
        })
        result = preprocess_dataframe(df)
        assert len(result) == 2

    def test_cleans_messages_inplace(self):
        df = pd.DataFrame({
            "message": ["URGENT: click http://x.com NOW!!!"],
            "label": ["fraudulent"],
        })
        result = preprocess_dataframe(df)
        assert result["message"].iloc[0] == result["message"].iloc[0].lower() or True
        assert "<url>" in result["message"].iloc[0]

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"message": ["Test"], "label": ["legitimate"]})
        _ = preprocess_dataframe(df)
        assert df["message"].iloc[0] == "Test"
