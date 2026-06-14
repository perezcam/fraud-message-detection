"""
Clasificador en cascada multi-capa.

Capas (en orden de ejecución):
  1. Reglas heurísticas   — siempre, 0 ms, sin API
  2. Clasificador ML       — siempre, ~5 ms, sin API
  3. Embeddings semánticos — condicional, ~300 ms, mistral-embed (barato)
  4. LLM                   — solo si ambiguo, ~1000 ms, mistral-small-latest

Gate 1 (tras capas 1+2): omite todo si ML y reglas coinciden con certeza alta.
Gate 2 (tras capa 3):    omite LLM si los embeddings confirman el veredicto ML.
"""

import logging
import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except ImportError:
    pass

from src.config import BAYES_NET_FILE, CASE_BASE_FILE, MODELS_DIR
from src.ml.predict import FraudPredictor
from src.ml.anomaly import AnomalyDetector, ANOMALY_FILE, _ANOMALY_THRESHOLD
from src.detection.meta_learner import CascadeMetaLearner, META_FILE, _extract_meta_features
from src.ml.transformer import TRANSFORMER_DIR

logger = logging.getLogger(__name__)

INDEX_PATH       = MODELS_DIR / "semantic_index.npz"
ANOMALY_PATH     = MODELS_DIR / ANOMALY_FILE
META_PATH        = MODELS_DIR / META_FILE
TRANSFORMER_PATH = MODELS_DIR / TRANSFORMER_DIR
BAYES_PATH       = MODELS_DIR / BAYES_NET_FILE
CASE_BASE_PATH   = MODELS_DIR / CASE_BASE_FILE

# ---------------------------------------------------------------------------
# Umbrales (valores por defecto — sobreescribibles vía CascadePredictor(cfg=…))
# ---------------------------------------------------------------------------
_ML_CONF_CERTAIN   = 0.95
_RULE_FRAUD_MIN    = 65
_RULE_LEGIT_MAX    = 15

_EMB_FRAUD_CONFIRM = 0.88
_EMB_LEGIT_CONFIRM = 0.88
_EMB_FRAUD_THREAT  = 0.80

_ANOMALY_BOOST_THR   = 0.75   # anomaly_score mínimo para elevar riesgo
_TRANSFORMER_HIGH    = 0.70   # transformer_proba → high
_TRANSFORMER_MED     = 0.50   # transformer_proba → medium
_BAYES_HIGH_THR      = 0.85   # bayes_score → medium boost
_CBR_HIGH_THR        = 0.86   # cbr_score (≥6/7) → medium boost
_META_HIGH_THR       = 0.70   # meta_proba → high
_META_MED_THR        = 0.35   # meta_proba → medium

# Claves reconocidas en cfg para validación
_CFG_KEYS = {
    "ml_conf_certain", "rule_fraud_min", "rule_legit_max",
    "emb_fraud_confirm", "emb_legit_confirm", "emb_fraud_threat",
    "anomaly_boost_thr", "transformer_high", "transformer_med",
    "bayes_high_thr", "cbr_high_thr", "meta_high_thr", "meta_med_thr",
    "risk_medium_threshold", "risk_high_threshold",
}

_RISK_ORDER = {"low": 0, "medium": 1, "high": 2}

_FRAUD_TYPE_SUFFIX = {
    "phishing":         " Posible intento de phishing.",
    "smishing":         " Es un posible SMS de smishing.",
    "prize_scam":       " Oferta de premio falso — no responda.",
    "urgency_scam":     " Usa urgencia artificial para presionarle.",
    "credential_theft": " Busca robar credenciales o códigos de verificación.",
    "financial_scam":   " Intenta obtener dinero o datos bancarios.",
    "spam":             " Mensaje de spam no solicitado.",
}


def _higher_risk(a: str, b: str) -> str:
    return a if _RISK_ORDER.get(a, 0) >= _RISK_ORDER.get(b, 0) else b


def _verdict_to_risk(verdict: str, confidence: float) -> str:
    if verdict == "fraudulent":
        return "high" if confidence >= 0.7 else "medium"
    if verdict == "suspicious":
        return "medium"
    return "low"


def _build_recommendation(risk_level: str, fraud_type: str = "none") -> str:
    base = {
        "low":    "No se detectan señales fuertes de fraude.",
        "medium": "El mensaje contiene señales sospechosas. Verifique la fuente antes de responder.",
        "high":   "Posible fraude. No comparta datos personales, códigos ni realice pagos.",
    }.get(risk_level, "Verifique el mensaje con precaución.")
    return base + _FRAUD_TYPE_SUFFIX.get(fraud_type, "")


class CascadePredictor:
    """
    Predictor en cascada de 4 capas para detección de mensajes fraudulentos.

    Parámetros:
        api_key:           API key de Mistral (o MISTRAL_API_KEY en .env).
        llm_model:         Modelo LLM (o MISTRAL_MODEL en .env).
        enable_llm:        Si False, opera sin LLM.
        enable_embeddings: Si False, opera sin embeddings.
        cfg:               Dict opcional con overrides de umbrales. Claves válidas:
                           ml_conf_certain, rule_fraud_min, rule_legit_max,
                           emb_fraud_confirm, emb_legit_confirm, emb_fraud_threat,
                           anomaly_boost_thr, transformer_high, transformer_med,
                           bayes_high_thr, cbr_high_thr, meta_high_thr, meta_med_thr.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        enable_llm: bool = True,
        enable_embeddings: bool = True,
        cfg: Optional[dict] = None,
    ) -> None:
        self._cfg: dict = cfg or {}
        if llm_model is None:
            llm_model = os.environ.get("MISTRAL_MODEL", "open-mistral-nemo")

        self._ml          = FraudPredictor()
        self._emb:        Optional[object] = None
        self._llm:        Optional[object] = None
        self._anomaly:    Optional[AnomalyDetector] = None
        self._meta:       Optional[CascadeMetaLearner] = None
        self._transformer: Optional[object] = None
        self._bayes:      Optional[object] = None
        self._case_base:  Optional[object] = None

        # Detector de anomalías (Isolation Forest)
        if ANOMALY_PATH.exists():
            try:
                det = AnomalyDetector()
                det.load(ANOMALY_PATH)
                self._anomaly = det
                logger.info("AnomalyDetector cargado.")
            except Exception as exc:
                logger.warning(f"AnomalyDetector no disponible: {exc}")
        else:
            logger.info(
                "AnomalyDetector no encontrado. "
                "Para activarlo: python main.py train-anomaly --dataset data/processed/messages.csv"
            )

        # Meta-learner de stacking
        if META_PATH.exists():
            try:
                meta = CascadeMetaLearner()
                meta.load(META_PATH)
                self._meta = meta
                logger.info("Meta-learner cargado.")
            except Exception as exc:
                logger.warning(f"Meta-learner no disponible: {exc}")
        else:
            logger.info(
                "Meta-learner no encontrado. "
                "Para activarlo: python main.py train-meta --dataset data/processed/messages.csv"
            )

        # Transformer XLM-RoBERTa (opcional, requiere fine-tuning previo)
        if TRANSFORMER_PATH.exists():
            try:
                from src.ml.transformer import TransformerFraudClassifier
                clf = TransformerFraudClassifier()
                clf.load(TRANSFORMER_PATH)
                self._transformer = clf
                logger.info("TransformerFraudClassifier cargado.")
            except Exception as exc:
                logger.warning(f"Transformer no disponible: {exc}")
        else:
            logger.info(
                "Transformer no encontrado. "
                "Para activarlo: python main.py train-transformer --dataset data/processed/messages.csv"
            )

        # Red Bayesiana (opcional, requiere train-bayes previo)
        if BAYES_PATH.exists():
            try:
                from src.ml.bayesian_net import FraudBayesNet
                bn = FraudBayesNet()
                bn.load(BAYES_PATH)
                self._bayes = bn
                logger.info("FraudBayesNet cargado.")
            except Exception as exc:
                logger.warning(f"BayesNet no disponible: {exc}")
        else:
            logger.info(
                "FraudBayesNet no encontrado. "
                "Para activarlo: python main.py train-bayes --dataset data/processed/messages.csv"
            )

        # CaseBase CBR (opcional, requiere build-cases previo)
        if CASE_BASE_PATH.exists():
            try:
                from src.detection.case_base import CaseBase
                from src.ml.features import load_vectorizer
                vectorizer = load_vectorizer()
                cb = CaseBase()
                cb.load(vectorizer, CASE_BASE_PATH)
                self._case_base = cb
                logger.info("CaseBase cargada.")
            except Exception as exc:
                logger.warning(f"CaseBase no disponible: {exc}")
        else:
            logger.info(
                "CaseBase no encontrada. "
                "Para activarla: python main.py build-cases --dataset data/processed/messages.csv"
            )

        if enable_embeddings:
            if INDEX_PATH.exists():
                try:
                    from src.llm.embeddings import SemanticIndex
                    emb = SemanticIndex(api_key=api_key)
                    emb.load()
                    self._emb = emb
                    logger.info("Índice semántico cargado.")
                except Exception as exc:
                    logger.warning(f"Embeddings no disponibles: {exc}")
            else:
                logger.info(
                    "Índice semántico no encontrado. "
                    "Para activarlo: python main.py build-index --dataset data/processed/messages.csv"
                )

        if enable_llm:
            try:
                from src.llm.analyzer import MistralFraudAnalyzer
                self._llm = MistralFraudAnalyzer(api_key=api_key, model=llm_model)
            except Exception as exc:
                logger.warning(f"LLM no disponible: {exc}")

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def predict(self, message: str) -> dict:
        ml_result  = self._ml.predict(message)
        ml_label   = ml_result["predicted_class"]
        ml_conf    = ml_result["confidence"] or 0.0
        rule_score = ml_result["risk_score"]
        layers     = ["rules", "ml"]

        # Capa Transformer XLM-RoBERTa (opcional, corre antes de embeddings)
        transformer_proba: Optional[float] = None
        if self._transformer is not None:
            try:
                transformer_proba = self._transformer.predict_proba(message)
                layers.append("transformer")
            except Exception as exc:
                logger.warning(f"Error en Transformer: {exc}")

        # Capa de anomalías: independiente de las demás capas
        anomaly_score: Optional[float] = None
        if self._anomaly is not None:
            try:
                anomaly_score = self._anomaly.score(message)
                layers.append("anomaly")
            except Exception as exc:
                logger.warning(f"Error en AnomalyDetector: {exc}")

        # Red Bayesiana (opcional)
        bayes_score: Optional[float] = None
        if self._bayes is not None:
            try:
                bayes_score = self._bayes.score_message(message)
                layers.append("bayes")
            except Exception as exc:
                logger.warning(f"Error en BayesNet: {exc}")

        # CaseBase CBR (opcional)
        cbr_score: Optional[float] = None
        if self._case_base is not None:
            try:
                cbr_result = self._case_base.query(message)
                cbr_score = cbr_result["cbr_score"]
                layers.append("cbr")
            except Exception as exc:
                logger.warning(f"Error en CaseBase: {exc}")

        if _gate1(ml_label, ml_conf, rule_score, self._cfg):
            return _aggregate(
                ml_result, None, None, layers, anomaly_score,
                transformer_proba, bayes_score, cbr_score, self._cfg,
            )

        emb_result: Optional[dict] = None
        if self._emb is not None:
            try:
                emb_result = self._emb.search(message)
                layers.append("embeddings")
            except Exception as exc:
                logger.warning(f"Error en búsqueda de embeddings: {exc}")

        if emb_result and _gate2(ml_label, ml_conf, emb_result, self._cfg):
            return _aggregate(
                ml_result, emb_result, None, layers, anomaly_score,
                transformer_proba, bayes_score, cbr_score, self._cfg,
            )

        llm_result: Optional[dict] = None
        if self._llm is not None:
            context = {
                "ml_label":      ml_label,
                "ml_confidence": round(ml_conf, 3),
                "rule_score":    rule_score,
                "signals":       ml_result["signals"],
            }
            if emb_result:
                context["fraud_similarity"] = emb_result["fraud_similarity"]
                context["legit_similarity"] = emb_result["legit_similarity"]
            if anomaly_score is not None:
                context["anomaly_score"] = round(anomaly_score, 3)
            if transformer_proba is not None:
                context["transformer_proba"] = round(transformer_proba, 3)
            if bayes_score is not None:
                context["bayes_score"] = round(bayes_score, 3)
            if cbr_score is not None:
                context["cbr_score"] = round(cbr_score, 3)
            llm_result = self._llm.analyze(message, context=context)
            layers.append("llm")

        result = _aggregate(
            ml_result, emb_result, llm_result, layers, anomaly_score,
            transformer_proba, bayes_score, cbr_score, self._cfg,
        )

        # Meta-learner: reemplaza el risk_level con la combinación aprendida
        if self._meta is not None:
            try:
                meta_feats = _extract_meta_features(self, message)
                meta_proba = self._meta.predict_proba_from_array(meta_feats)
                meta_high = self._cfg.get("meta_high_thr", _META_HIGH_THR)
                meta_med  = self._cfg.get("meta_med_thr",  _META_MED_THR)
                if meta_proba >= meta_high:
                    meta_risk = "high"
                elif meta_proba >= meta_med:
                    meta_risk = "medium"
                else:
                    meta_risk = "low"
                result["risk_level"]     = _higher_risk(result["risk_level"], meta_risk)
                result["meta_proba"]     = round(meta_proba, 4)
                result["layers_used"]    = result.get("layers_used", layers) + ["meta"]
                result["recommendation"] = _build_recommendation(result["risk_level"])
            except Exception as exc:
                logger.warning(f"Meta-learner inference falló: {exc}")

        return result

    def predict_batch(self, messages: list[str]) -> list[dict]:
        return [self.predict(msg) for msg in messages]


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def _gate1(ml_label: str, ml_conf: float, rule_score: int, cfg: dict = {}) -> bool:
    ml_conf_certain = cfg.get("ml_conf_certain", _ML_CONF_CERTAIN)
    rule_fraud_min  = cfg.get("rule_fraud_min",  _RULE_FRAUD_MIN)
    rule_legit_max  = cfg.get("rule_legit_max",  _RULE_LEGIT_MAX)
    if ml_label == "fraudulent" and ml_conf >= ml_conf_certain and rule_score >= rule_fraud_min:
        return True
    if ml_label == "legitimate" and ml_conf >= ml_conf_certain and rule_score < rule_legit_max:
        return True
    return False


def _gate2(ml_label: str, ml_conf: float, emb: dict, cfg: dict = {}) -> bool:
    fraud_sim = emb["fraud_similarity"]
    legit_sim = emb["legit_similarity"]
    emb_fraud_confirm = cfg.get("emb_fraud_confirm", _EMB_FRAUD_CONFIRM)
    emb_legit_confirm = cfg.get("emb_legit_confirm", _EMB_LEGIT_CONFIRM)
    emb_fraud_threat  = cfg.get("emb_fraud_threat",  _EMB_FRAUD_THREAT)
    if ml_label == "fraudulent" and fraud_sim >= emb_fraud_confirm:
        return True
    if ml_label == "legitimate" and legit_sim >= emb_legit_confirm and fraud_sim < emb_fraud_threat:
        return True
    return False


# ---------------------------------------------------------------------------
# Agregación
# ---------------------------------------------------------------------------

def _aggregate(
    ml_result: dict,
    emb_result: Optional[dict],
    llm_result: Optional[dict],
    layers: list[str],
    anomaly_score: Optional[float] = None,
    transformer_proba: Optional[float] = None,
    bayes_score: Optional[float] = None,
    cbr_score: Optional[float] = None,
    cfg: dict = {},
) -> dict:
    base = {
        "original_message":      ml_result["original_message"],
        "preprocessed_message":  ml_result["preprocessed_message"],
        "risk_score":            ml_result["risk_score"],
        "signals":               ml_result["signals"],
        "layers_used":           layers,
        "embedding_result":      emb_result,
        "anomaly_score":         anomaly_score,
        "transformer_proba":     transformer_proba,
        "bayes_score":           bayes_score,
        "cbr_score":             cbr_score,
    }

    if llm_result and "error" not in llm_result:
        llm_risk   = _verdict_to_risk(llm_result["verdict"], llm_result["confidence"])
        final_risk = _higher_risk(ml_result["risk_level"], llm_risk)
        base.update({
            "predicted_class": llm_result["verdict"],
            "confidence":      llm_result["confidence"],
            "risk_level":      final_risk,
            "recommendation":  _build_recommendation(final_risk, llm_result.get("fraud_type", "none")),
            "llm_analysis": {
                "fraud_type":  llm_result.get("fraud_type", "none"),
                "indicators":  llm_result.get("indicators", []),
                "explanation": llm_result.get("explanation", ""),
            },
        })
    else:
        emb_boost = None
        if emb_result:
            fsim = emb_result["fraud_similarity"]
            if fsim >= 0.85:
                emb_boost = "high"
            elif fsim >= 0.70:
                emb_boost = "medium"
        final_risk = _higher_risk(ml_result["risk_level"], emb_boost) if emb_boost else ml_result["risk_level"]

        anomaly_boost_thr = cfg.get("anomaly_boost_thr", _ANOMALY_BOOST_THR)
        transformer_high  = cfg.get("transformer_high",  _TRANSFORMER_HIGH)
        transformer_med   = cfg.get("transformer_med",   _TRANSFORMER_MED)
        bayes_high_thr    = cfg.get("bayes_high_thr",    _BAYES_HIGH_THR)
        cbr_high_thr      = cfg.get("cbr_high_thr",      _CBR_HIGH_THR)

        # Anomaly boost
        if (
            anomaly_score is not None
            and anomaly_score >= anomaly_boost_thr
            and ml_result["predicted_class"] == "legitimate"
        ):
            final_risk = _higher_risk(final_risk, "medium")

        # Transformer boost
        if transformer_proba is not None:
            if transformer_proba >= transformer_high:
                final_risk = _higher_risk(final_risk, "high")
            elif transformer_proba >= transformer_med:
                final_risk = _higher_risk(final_risk, "medium")

        # Bayes boost
        if bayes_score is not None and bayes_score >= bayes_high_thr:
            final_risk = _higher_risk(final_risk, "medium")

        # CBR boost
        if (
            cbr_score is not None
            and cbr_score >= cbr_high_thr
            and ml_result["predicted_class"] != "legitimate"
        ):
            final_risk = _higher_risk(final_risk, "medium")

        base.update({
            "predicted_class": ml_result["predicted_class"],
            "confidence":      ml_result["confidence"],
            "risk_level":      final_risk,
            "recommendation":  _build_recommendation(final_risk),
            "llm_analysis":    None,
        })

    return base
