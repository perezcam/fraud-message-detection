"""
Módulo de configuración centralizada del proyecto.
Define rutas, nombres de columnas, etiquetas, y parámetros de entrenamiento.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas base
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LOCAL_VALIDATION_DIR = DATA_DIR / "local_validation"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

# Crear directorios si no existen
for _dir in [RAW_DATA_DIR, PROCESSED_DATA_DIR, LOCAL_VALIDATION_DIR,
             MODELS_DIR, FIGURES_DIR, METRICS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Nombres estándar de columnas
# ---------------------------------------------------------------------------
TEXT_COLUMN = "message"
LABEL_COLUMN = "label"

# Alias posibles para la columna de texto
TEXT_COLUMN_ALIASES = ["text", "sms", "message", "body", "email", "content", "msg"]

# Alias posibles para la columna de etiqueta
LABEL_COLUMN_ALIASES = ["label", "category", "target", "class", "type", "v1", "Tag"]

# ---------------------------------------------------------------------------
# Mapeo de etiquetas a categorías normalizadas
# ---------------------------------------------------------------------------
LABEL_MAPPING: dict[str, str] = {
    # Legítimo
    "ham": "legitimate",
    "legitimate": "legitimate",
    "safe": "legitimate",
    "normal": "legitimate",
    "ok": "legitimate",
    "benign": "legitimate",
    "not spam": "legitimate",
    "not_spam": "legitimate",
    # Fraudulento
    "spam": "fraudulent",
    "smishing": "fraudulent",
    "phishing": "fraudulent",
    "fraud": "fraudulent",
    "scam": "fraudulent",
    "malicious": "fraudulent",
    "phish": "fraudulent",
    "fraudulent": "fraudulent",
    "smish": "fraudulent",
    # Sospechoso
    "suspicious": "suspicious",
    "warning": "suspicious",
    "unknown": "suspicious",
}

# Etiquetas normalizadas válidas
VALID_LABELS = ["legitimate", "suspicious", "fraudulent"]

# Etiquetas para clasificación binaria (primera versión)
BINARY_LABELS = ["legitimate", "fraudulent"]

# ---------------------------------------------------------------------------
# Parámetros de entrenamiento
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ---------------------------------------------------------------------------
# Parámetros TF-IDF
# ---------------------------------------------------------------------------
TFIDF_MAX_FEATURES = 10_000
TFIDF_NGRAM_RANGE = (1, 3)
TFIDF_MIN_DF = 2

# ---------------------------------------------------------------------------
# Negación — palabras que invierten el significado de una señal de fraude
# ---------------------------------------------------------------------------
NEGATION_WORDS = [
    "no", "not", "never", "jamás", "nunca", "sin", "without",
    "ningún", "ninguna", "ninguno", "nada", "nadie",
    "don't", "doesn't", "didn't", "won't", "wouldn't", "can't",
]

# ---------------------------------------------------------------------------
# Nombres de archivos — componentes nuevos
# ---------------------------------------------------------------------------
BAYES_NET_FILE          = "bayes_net.joblib"
CASE_BASE_FILE          = "case_base.npz"
OPTIMAL_THRESHOLDS_FILE = "optimal_thresholds.json"
PSO_HYPERPARAMS_FILE    = "pso_hyperparams.json"
EDA_GENERATOR_FILE      = "eda_generator.joblib"
TABU_THRESHOLDS_FILE    = "tabu_thresholds.json"

# ---------------------------------------------------------------------------
# Modelos soportados
# ---------------------------------------------------------------------------
SUPPORTED_MODELS = [
    "naive_bayes",
    "logistic_regression",
    "linear_svc",
    "linear_svc_calibrated",
    "random_forest",
    "xgboost",
    "lightgbm",
]
DEFAULT_MODEL = "logistic_regression"

# ---------------------------------------------------------------------------
# Nombres de archivos de artefactos
# ---------------------------------------------------------------------------
PROCESSED_DATASET_NAME = "messages.csv"
MODEL_FILE_TEMPLATE = "{model_name}_model.joblib"
VECTORIZER_FILE_NAME = "tfidf_vectorizer.joblib"
METADATA_FILE_NAME = "experiment_metadata.json"

# ---------------------------------------------------------------------------
# Umbrales de riesgo (puntaje 0-100)
# ---------------------------------------------------------------------------
RISK_SCORE_MEDIUM_THRESHOLD = 30
RISK_SCORE_HIGH_THRESHOLD = 60

# ---------------------------------------------------------------------------
# Vocabularios de señales sospechosas (español e inglés)
# ---------------------------------------------------------------------------
URGENCY_WORDS = [
    "urgente", "urgency", "urgent", "inmediatamente", "immediately",
    "ahora mismo", "right now", "ahora", "now", "rápido", "quickly",
    "expira", "expires", "vence", "deadline", "límite", "ultimo", "último",
    "final", "suspendido", "suspended", "bloqueado", "blocked",
    "cancelado", "cancelled", "verificar", "verify", "confirmar", "confirm",
    "actualizar", "update", "acceso restringido", "restricted access",
    "inmediato", "immediate",
]

MONEY_WORDS = [
    "dinero", "money", "pesos", "dólares", "dollars", "euros",
    "transferencia", "transfer", "depósito", "deposit", "pago", "payment",
    "cobro", "charge", "cuenta", "account", "banco", "bank",
    "tarjeta", "card", "crédito", "credit", "débito", "debit",
    "premio", "prize", "reward", "recompensa", "ganó", "won",
    "gratis", "free", "oferta", "offer", "descuento", "discount",
    "remesa", "remittance", "efectivo", "cash",
]

CREDENTIAL_WORDS = [
    "contraseña", "password", "clave", "key", "pin", "otp", "nip",
    "código", "code", "token", "número secreto", "secret number",
    "usuario", "user", "username", "datos personales", "personal data",
    "información confidencial", "confidential", "credencial", "credential",
    "acceso", "access", "login", "sesión", "session",
    "verificación", "verification", "autenticación", "authentication",
]

PRIZE_WORDS = [
    "premio", "prize", "ganó", "won", "felicidades", "congratulations",
    "ganador", "winner", "seleccionado", "selected", "elegido", "chosen",
    "sorteo", "lottery", "rifa", "raffle", "regalo", "gift",
    "oferta exclusiva", "exclusive offer", "descuento especial",
    "promoción", "promotion", "gratis", "free", "obsequio",
    "beneficio", "benefit", "recompensa", "reward",
]
