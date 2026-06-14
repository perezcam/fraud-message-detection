"""
Modelo neuronal para clasificación de secuencias conversacionales.

Arquitectura:
  1. RiskFeatureEncoder
       Convierte cada mensaje en un vector de 10 features de riesgo computadas
       por el motor de reglas (analyze_risk).  Es completamente agnóstico al
       idioma y no requiere entrenamiento previo.
       Dimensión de salida: EMBED_DIM = 10.

  2. ConversationAttentionLSTM (PyTorch)
       - Bidirectional LSTM (2 capas) sobre la secuencia de feature-vectors
         → captura dependencias temporales hacia adelante y atrás
       - Mecanismo de atención tipo dot-product sobre las salidas del LSTM
         → pondera cada mensaje por su relevancia para la clasificación
         → las attention weights son interpretables (qué mensaje es clave)
       - Clasificador final: Linear → ReLU → Dropout → Linear

  3. NeuralConversationClassifier
       Wrapper de alto nivel con fit() / predict() / save() / load().
       Genera secuencias sintéticas del dataset SMS para entrenar.

Por qué este enfoque es fuerte:
  - El LSTM genuinamente "ve" la secuencia en orden (el orden importa)
  - La atención aprende DÓNDE está la señal de fraude en la conversación
  - Las features de riesgo son agnósticas al idioma → funciona en español
  - La feature risk_delta captura escalada (confianza → ataque)
  - Entrenado con backpropagation sobre gradientes reales, no sobre reglas
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

EMBED_DIM   = 10   # dimensión del vector de features de riesgo por mensaje
HIDDEN_DIM  = 128  # dimensión LSTM por dirección (256 total en BiLSTM)
N_LAYERS    = 2
MAX_SEQ     = 10   # máximo de mensajes por ventana (pad / truncar)
NEURAL_FILE = "conversation_neural.pt"


# ---------------------------------------------------------------------------
# Encoder: features de riesgo por mensaje (agnóstico al idioma)
# ---------------------------------------------------------------------------

class RiskFeatureEncoder:
    """
    Convierte cada Message en un vector de features de riesgo (EMBED_DIM dims).

    Usa el dict individual_risk ya computado por analyze_risk —
    no requiere entrenamiento ni llamadas a API.

    Features (10 dims):
        0  risk_score_norm   — risk_score / 100
        1  risk_level_num    — 0.0=low  0.5=medium  1.0=high
        2  has_credential    — solicita contraseña/NIP/OTP
        3  has_urgency       — lenguaje urgente
        4  has_pressure      — amenaza de bloqueo / presión
        5  has_transfer      — solicita transferencia/pago
        6  has_prize         — menciona premios/regalos
        7  has_financial     — menciona términos financieros/montos
        8  text_len          — len(text) / 300, cap 1.0
        9  risk_delta        — max(risk_score - prev_risk, 0) / 100
    """

    FEATURE_DIM = EMBED_DIM
    _LEVEL = {"low": 0.0, "medium": 0.5, "high": 1.0, "critical": 1.0}

    _SIG_CREDENTIAL = "Solicita contraseña"
    _SIG_URGENCY    = "Usa lenguaje de urgencia"
    _SIG_PRESSURE   = "presión"    # también cubre "Amenaza con bloqueo"
    _SIG_BLOCK      = "bloqueo"
    _SIG_TRANSFER   = "Solicita transferencia"
    _SIG_PRIZE      = "Menciona premios"
    _SIG_FINANCIAL  = "financieros"    # cubre "términos financieros" y "cantidades monetarias"

    def __init__(self):
        self._n_components = self.FEATURE_DIM
        self._fitted = True  # sin entrenamiento necesario

    def _flags(self, signals: list[str]) -> tuple[float, ...]:
        joined = " ".join(signals).lower()
        return (
            float(self._SIG_CREDENTIAL.lower() in joined),
            float(self._SIG_URGENCY.lower()    in joined),
            float(self._SIG_PRESSURE           in joined or self._SIG_BLOCK in joined),
            float(self._SIG_TRANSFER.lower()   in joined),
            float(self._SIG_PRIZE.lower()      in joined),
            float(self._SIG_FINANCIAL          in joined or "monetaria" in joined),
        )

    def encode_message(self, msg, prev_risk_norm: float = 0.0) -> np.ndarray:
        """Devuelve vector de 10 features para un Message con individual_risk ya poblado."""
        r       = msg.individual_risk or {}
        score   = r.get("risk_score", 0) / 100.0
        level   = self._LEVEL.get(r.get("risk_level", "low"), 0.0)
        signals = r.get("signals", [])
        cred, urg, press, trans, prize, fin = self._flags(signals)
        text_len = min(len(msg.text) / 300.0, 1.0)
        delta    = max(score - prev_risk_norm, 0.0)

        return np.array(
            [score, level, cred, urg, press, trans, prize, fin, text_len, delta],
            dtype=np.float32,
        )

    def encode_sequence(
        self, messages: list, max_len: int = MAX_SEQ
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Devuelve (features_padded (max_len, EMBED_DIM), mask (max_len,)).
        mask[i] = True  →  posición de padding, ignorar en atención.
        """
        n      = min(len(messages), max_len)
        padded = np.zeros((max_len, self.FEATURE_DIM), dtype=np.float32)
        prev   = 0.0
        for i, msg in enumerate(messages[:n]):
            feat    = self.encode_message(msg, prev_risk_norm=prev)
            padded[i] = feat
            prev    = feat[0]  # risk_score para el siguiente delta
        mask = torch.zeros(max_len, dtype=torch.bool)
        mask[n:] = True
        return torch.from_numpy(padded), mask


# ---------------------------------------------------------------------------
# Modelo: Bidirectional LSTM + Atención
# ---------------------------------------------------------------------------

class ConversationAttentionLSTM(nn.Module):
    """
    Clasificador secuencial con BiLSTM y mecanismo de atención.

    Input :  tensor (batch, seq_len, EMBED_DIM)
    Output:  (logits (batch,), attention_weights (batch, seq_len))

    IMPORTANTE: forward() devuelve LOGITS crudos (sin sigmoid).
    Aplicar sigmoid externamente al hacer inferencia.
    El entrenamiento usa BCEWithLogitsLoss directamente sobre los logits.
    """

    def __init__(
        self,
        input_dim:  int   = EMBED_DIM,
        hidden_dim: int   = HIDDEN_DIM,
        n_layers:   int   = N_LAYERS,
        dropout:    float = 0.35,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden_dim * 2  # bidireccional

        self.attn = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.Tanh(),
            nn.Linear(lstm_out_dim // 2, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        x:    torch.Tensor,                    # (batch, seq_len, input_dim)
        mask: Optional[torch.Tensor] = None,   # (batch, seq_len) — True = padding
    ) -> tuple[torch.Tensor, torch.Tensor]:

        lstm_out, _ = self.lstm(x)             # (batch, seq_len, hidden*2)

        scores = self.attn(lstm_out)           # (batch, seq_len, 1)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-1), float("-inf"))
        attn_weights = torch.softmax(scores, dim=1)   # (batch, seq_len, 1)

        context = (attn_weights * lstm_out).sum(dim=1)  # (batch, hidden*2)
        logits  = self.classifier(context).squeeze(-1)  # (batch,)  — sin sigmoid

        return logits, attn_weights.squeeze(-1)


# ---------------------------------------------------------------------------
# Wrapper de alto nivel
# ---------------------------------------------------------------------------

class NeuralConversationClassifier:
    """
    Clasificador neuronal de conversaciones basado en BiLSTM + atención.

    Uso:
        clf = NeuralConversationClassifier()
        clf.fit(df)
        clf.save()
        clf.load()
        prob, attn = clf.predict(messages)  # prob ∈ [0, 1]
    """

    _SEQ_TYPES = {
        "all_legit":    0.25,
        "all_fraud":    0.25,
        "trust_attack": 0.35,
        "escalation":   0.15,
    }

    def __init__(self) -> None:
        self._encoder: Optional[RiskFeatureEncoder] = None
        self._model:   Optional[ConversationAttentionLSTM] = None
        self._device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fitted   = False

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------

    def fit(
        self,
        df,
        n_synthetic:  int   = 5000,
        seq_length:   int   = 5,
        epochs:       int   = 80,
        batch_size:   int   = 64,
        lr:           float = 1e-3,
        random_state: int   = 42,
    ) -> None:
        """
        Genera secuencias sintéticas con features de riesgo reales,
        entrena el encoder de riesgo y el BiLSTM.

        Args:
            df:           DataFrame con columnas "message" y "label".
            n_synthetic:  Total de secuencias a generar.
            seq_length:   Mensajes por secuencia de entrenamiento.
            epochs:       Épocas de entrenamiento.
            batch_size:   Tamaño del mini-batch.
            lr:           Tasa de aprendizaje.
            random_state: Semilla para reproducibilidad.
        """
        from src.rules.risk import analyze_risk
        from src.conversation.models import Message

        torch.manual_seed(random_state)
        rng = np.random.default_rng(random_state)

        fraud_texts = df[df["label"] == "fraudulent"]["message"].dropna().tolist()
        legit_texts = df[df["label"] == "legitimate"]["message"].dropna().tolist()

        if not fraud_texts or not legit_texts:
            raise ValueError("El DataFrame debe contener mensajes de ambas clases.")

        # Inicializar encoder (no requiere fit)
        encoder = RiskFeatureEncoder()
        self._encoder = encoder

        # Pre-calcular features de riesgo para el pool completo
        logger.info("Computando features de riesgo para los pools de mensajes...")
        cap = 3000  # cap para que el pre-cómputo sea rápido

        def _enrich(texts: list[str]) -> list:
            pool = []
            for t in texts[:cap]:
                m = Message(text=t)
                m.individual_risk = analyze_risk(t)
                pool.append(m)
            return pool

        fraud_pool = _enrich(fraud_texts)
        legit_pool = _enrich(legit_texts)

        def _sample(pool: list, size: int) -> list:
            idx = rng.integers(0, len(pool), size=size)
            return [pool[i] for i in idx]

        logger.info(
            f"Generando {n_synthetic} secuencias sintéticas "
            f"(seq_length={seq_length}, device={self._device})."
        )

        counts = {t: max(1, int(n_synthetic * w)) for t, w in self._SEQ_TYPES.items()}

        X_seqs:  list[torch.Tensor] = []
        X_masks: list[torch.Tensor] = []
        y_list:  list[float] = []

        # --- all_legit: secuencias completamente legítimas → label 0 ---
        for _ in range(counts["all_legit"]):
            msgs = _sample(legit_pool, seq_length)
            seq, mask = encoder.encode_sequence(msgs, MAX_SEQ)
            X_seqs.append(seq); X_masks.append(mask); y_list.append(0.0)

        # --- all_fraud: secuencias completamente fraudulentas → label 1 ---
        for _ in range(counts["all_fraud"]):
            msgs = _sample(fraud_pool, seq_length)
            seq, mask = encoder.encode_sequence(msgs, MAX_SEQ)
            X_seqs.append(seq); X_masks.append(mask); y_list.append(1.0)

        # --- trust_attack: inicio legítimo, final de ataque → label 1 ---
        # Este tipo replica el patrón BBVA: establecer confianza → atacar
        for _ in range(counts["trust_attack"]):
            n_legit = max(1, int(seq_length * rng.uniform(0.40, 0.70)))
            n_fraud = max(1, seq_length - n_legit)
            msgs    = _sample(legit_pool, n_legit) + _sample(fraud_pool, n_fraud)
            seq, mask = encoder.encode_sequence(msgs, MAX_SEQ)
            X_seqs.append(seq); X_masks.append(mask); y_list.append(1.0)

        # --- escalation: riesgo creciente → label 1 ---
        for _ in range(counts["escalation"]):
            candidates = _sample(legit_pool + fraud_pool, seq_length * 4)
            candidates.sort(
                key=lambda m: m.individual_risk.get("risk_score", 0) if m.individual_risk else 0
            )
            # Tomar la mitad inferior como inicio y la superior como final
            half = max(2, seq_length // 2)
            early_msgs = candidates[: len(candidates) // 2]
            late_msgs  = candidates[len(candidates) // 2 :]
            early = list(rng.choice(len(early_msgs), size=seq_length - half, replace=False))
            late  = list(rng.choice(len(late_msgs),  size=half,             replace=False))
            msgs  = [early_msgs[i] for i in early] + [late_msgs[i] for i in late]
            seq, mask = encoder.encode_sequence(msgs, MAX_SEQ)
            X_seqs.append(seq); X_masks.append(mask); y_list.append(1.0)

        X = torch.stack(X_seqs).to(self._device)
        M = torch.stack(X_masks).to(self._device)
        y = torch.tensor(y_list, dtype=torch.float32).to(self._device)

        fraud_n = int(y.sum().item())
        legit_n = len(y) - fraud_n
        logger.info(
            f"Dataset sintético: {len(y)} secuencias "
            f"({fraud_n} sospechosas, {legit_n} legítimas)."
        )

        # --- Modelo ---
        model = ConversationAttentionLSTM(input_dim=EMBED_DIM).to(self._device)

        optimizer  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        pos_weight = torch.tensor(legit_n / max(fraud_n, 1), dtype=torch.float32).to(self._device)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # --- Entrenamiento ---
        N          = len(y)
        best_loss  = float("inf")
        patience   = 12
        no_improve = 0

        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(N)
            X_shuf, M_shuf, y_shuf = X[perm], M[perm], y[perm]

            epoch_loss = 0.0
            for i in range(0, N, batch_size):
                xb = X_shuf[i : i + batch_size]
                mb = M_shuf[i : i + batch_size]
                yb = y_shuf[i : i + batch_size]

                optimizer.zero_grad()
                logits, _ = model(xb, mb)      # logits crudos
                loss      = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(yb)

            epoch_loss /= N
            scheduler.step(epoch_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(f"  Época {epoch+1:3d}/{epochs} — loss={epoch_loss:.4f}")

            if epoch_loss < best_loss - 1e-4:
                best_loss  = epoch_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(f"  Early stopping en época {epoch+1}.")
                    break

        # Accuracy final
        model.eval()
        with torch.no_grad():
            logits_all, _ = model(X, M)
            preds = (torch.sigmoid(logits_all) > 0.5).float()
            acc   = (preds == y).float().mean().item()
        logger.info(f"Entrenamiento completado — loss_final={best_loss:.4f}, acc={acc:.2%}")

        self._model  = model
        self._fitted = True

    # ------------------------------------------------------------------
    # Inferencia
    # ------------------------------------------------------------------

    def predict(
        self, messages: list
    ) -> tuple[float, list[float]]:
        """
        Devuelve (prob_sospechoso, attention_weights_por_mensaje).

        Los attention_weights indican qué mensajes fueron más determinantes
        en la clasificación (suma = 1.0 en los mensajes no-padding).
        """
        if not self._fitted:
            return -1.0, []

        seq, mask = self._encoder.encode_sequence(messages, MAX_SEQ)
        seq  = seq.unsqueeze(0).to(self._device)   # (1, MAX_SEQ, EMBED_DIM)
        mask = mask.unsqueeze(0).to(self._device)

        self._model.eval()
        with torch.no_grad():
            logits, attn = self._model(seq, mask)

        prob = float(torch.sigmoid(logits[0]).item())
        n    = min(len(messages), MAX_SEQ)
        attn_list = attn[0, :n].cpu().numpy().tolist()

        return round(prob, 4), attn_list

    def predict_proba(self, messages) -> float:
        prob, _ = self.predict(messages)
        return prob

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> Path:
        import joblib
        from src.config import MODELS_DIR

        out = Path(path) if path else MODELS_DIR / NEURAL_FILE
        joblib.dump(
            {
                "model_state":  self._model.state_dict(),
                "model_config": {
                    "input_dim":  EMBED_DIM,
                    "hidden_dim": HIDDEN_DIM,
                    "n_layers":   N_LAYERS,
                },
                "encoder_type": "RiskFeatureEncoder",
            },
            out,
        )
        logger.info(f"Modelo neuronal guardado en {out}")
        return out

    def load(self, path: Optional[Path] = None) -> None:
        import joblib
        from src.config import MODELS_DIR

        p = Path(path) if path else MODELS_DIR / NEURAL_FILE
        if not p.exists():
            raise FileNotFoundError(f"Modelo neuronal no encontrado: {p}")

        data = joblib.load(p)
        cfg  = data["model_config"]

        model = ConversationAttentionLSTM(
            input_dim  = cfg["input_dim"],
            hidden_dim = cfg["hidden_dim"],
            n_layers   = cfg["n_layers"],
        ).to(self._device)
        model.load_state_dict(data["model_state"])
        model.eval()

        self._encoder = RiskFeatureEncoder()
        self._model   = model
        self._fitted  = True
        logger.info(f"Modelo neuronal cargado desde {p}")
