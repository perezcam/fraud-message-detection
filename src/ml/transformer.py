"""
Clasificador de fraude basado en XLM-RoBERTa fine-tuneado.

Por qué XLM-RoBERTa:
  - Modelo multilingüe preentrenado en 100 idiomas (inglés + español nativos)
  - Captura semántica profunda que TF-IDF no puede: "no es una estafa" ≠ "es una estafa"
  - Fine-tuning con 3-5 épocas es suficiente para clasificación binaria
  - Funciona en CPU (lento pero funcional para inferencia)

Arquitectura:
  XLM-RoBERTa-base → pooler del token [CLS] → Dropout(0.3) → Linear(768, 2)

Integración en cascade.py:
  Si models/transformer/ existe, CascadePredictor lo carga y usa como
  capa adicional cuyo predict_proba se incluye en los meta-features.

Guardado en models/transformer/ (directorio con config + weights de HuggingFace)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

TRANSFORMER_DIR  = "transformer"
BASE_MODEL       = "xlm-roberta-base"
MAX_LEN          = 128
_LABEL_MAP       = {"fraudulent": 1, "legitimate": 0}
_INT_TO_LABEL    = {1: "fraudulent", 0: "legitimate"}


class TransformerFraudClassifier:
    """
    Clasificador de fraude con XLM-RoBERTa fine-tuneado.

    Retorna el mismo formato de dict que FraudPredictor.predict() para
    compatibilidad con el resto del sistema.

    Uso:
        clf = TransformerFraudClassifier()
        metrics = clf.fit(df, epochs=3)
        clf.save()

        # Inferencia (compatible con FraudPredictor)
        result = clf.predict("Su cuenta será bloqueada...")
        proba  = clf.predict_proba("Su cuenta será bloqueada...")
    """

    def __init__(self) -> None:
        self._model     = None
        self._tokenizer = None
        self._fitted    = False
        self._device    = None

    def _get_device(self):
        try:
            import torch
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            return "cpu"

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------

    def fit(
        self,
        df,
        epochs:     int   = 3,
        batch_size: int   = 16,
        lr:         float = 2e-5,
        max_len:    int   = MAX_LEN,
        warmup_ratio: float = 0.1,
    ) -> dict:
        """
        Fine-tunea XLM-RoBERTa sobre el dataset procesado.

        Args:
            df:         DataFrame con columnas "message" y "label".
            epochs:     Épocas de fine-tuning (3-5 es suficiente).
            batch_size: Tamaño de batch (16 para CPU, 32+ para GPU).
            lr:         Learning rate (2e-5 es el valor canónico para BERT/RoBERTa).
            max_len:    Longitud máxima de tokens.
            warmup_ratio: Fracción de steps con warmup lineal.

        Returns:
            dict con accuracy y f1_fraudulent en el split de prueba.
        """
        try:
            import torch
            from torch.utils.data import DataLoader, WeightedRandomSampler
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                get_linear_schedule_with_warmup,
            )
        except ImportError as exc:
            raise ImportError("pip install transformers torch") from exc

        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score
        from src.config import LABEL_COLUMN, RANDOM_STATE, TEXT_COLUMN
        from src.data.preprocessing import clean_text

        device = self._get_device()
        self._device = device
        logger.info(f"Fine-tuning XLM-RoBERTa en {device}...")

        # Preparar datos
        df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
        df = df[df[LABEL_COLUMN].isin(_LABEL_MAP)]
        texts  = df[TEXT_COLUMN].astype(str).apply(clean_text).tolist()
        labels = df[LABEL_COLUMN].map(_LABEL_MAP).tolist()

        X_tr, X_te, y_tr, y_te = train_test_split(
            texts, labels,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=labels,
        )

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self._tokenizer = tokenizer

        def _tokenize(text_list):
            return tokenizer(
                text_list,
                max_length=max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        # Dataset simple (tensores en memoria)
        class _DS(torch.utils.data.Dataset):
            def __init__(self, texts, labels):
                enc = tokenizer(texts, max_length=max_len, padding="max_length",
                                truncation=True, return_tensors="pt")
                self.input_ids      = enc["input_ids"]
                self.attention_mask = enc["attention_mask"]
                self.labels         = torch.tensor(labels, dtype=torch.long)

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                return {
                    "input_ids":      self.input_ids[idx],
                    "attention_mask": self.attention_mask[idx],
                    "labels":         self.labels[idx],
                }

        train_ds = _DS(X_tr, y_tr)
        test_ds  = _DS(X_te, y_te)

        # Sampler con balanceo de clases
        class_counts = np.bincount(y_tr)
        weights = [1.0 / class_counts[y] for y in y_tr]
        sampler = WeightedRandomSampler(weights, len(weights))

        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size * 2)

        model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL, num_labels=2
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        total_steps   = len(train_loader) * epochs
        warmup_steps  = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        best_f1    = 0.0
        best_state = None
        patience   = 2
        no_improve = 0

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()

            # Evaluación al final de cada época
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in test_loader:
                    batch  = {k: v.to(device) for k, v in batch.items()}
                    logits = model(**batch).logits
                    preds  = logits.argmax(dim=-1).cpu().numpy()
                    lbls   = batch["labels"].cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(lbls)

            f1 = f1_score(all_labels, all_preds, average="binary", pos_label=1, zero_division=0)
            acc = accuracy_score(all_labels, all_preds)
            logger.info(
                f"  Época {epoch+1}/{epochs} — loss={epoch_loss/len(train_loader):.4f} "
                f"| acc={acc:.4f} | f1_fraud={f1:.4f}"
            )

            if f1 > best_f1 + 1e-4:
                best_f1    = f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(f"  Early stopping en época {epoch+1}.")
                    break

        if best_state:
            model.load_state_dict(best_state)

        self._model  = model.to(device)
        self._fitted = True

        # Métricas finales
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch  = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                preds  = logits.argmax(dim=-1).cpu().numpy()
                lbls   = batch["labels"].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(lbls)

        final_acc = accuracy_score(all_labels, all_preds)
        final_f1  = f1_score(all_labels, all_preds, average="binary", pos_label=1, zero_division=0)
        logger.info(
            f"Fine-tuning completado — accuracy={final_acc:.4f}, f1_fraud={final_f1:.4f}"
        )
        return {"accuracy": round(final_acc, 4), "f1_fraudulent": round(final_f1, 4)}

    # ------------------------------------------------------------------
    # Inferencia
    # ------------------------------------------------------------------

    def predict_proba(self, message: str) -> float:
        """Retorna probabilidad de que el mensaje sea fraudulento (0-1)."""
        if not self._fitted:
            raise RuntimeError("TransformerFraudClassifier no entrenado.")
        import torch
        from src.data.preprocessing import clean_text

        clean = clean_text(message) or message
        enc = self._tokenizer(
            clean, max_length=MAX_LEN, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}
        self._model.eval()
        with torch.no_grad():
            logits = self._model(**enc).logits
            probs  = torch.softmax(logits, dim=-1)[0]
        return round(float(probs[1].item()), 4)   # índice 1 = fraudulent

    def predict(self, message: str) -> dict:
        """Retorna dict compatible con FraudPredictor.predict()."""
        from src.rules.risk import analyze_risk
        from src.ml.predict import _get_recommendation

        proba = self.predict_proba(message)
        label = "fraudulent" if proba >= 0.5 else "legitimate"

        if label == "fraudulent":
            risk_level = "high" if proba >= 0.70 else "medium"
        else:
            risk_level = "low" if proba < 0.35 else "medium"

        risk_info = analyze_risk(message)

        return {
            "original_message":     message,
            "preprocessed_message": message,
            "predicted_class":      label,
            "confidence":           proba,
            "risk_level":           risk_level,
            "risk_score":           risk_info["risk_score"],
            "signals":              risk_info["signals"],
            "recommendation":       _get_recommendation(risk_level),
            "model_type":           "transformer",
        }

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, directory: Optional[Path] = None) -> Path:
        """Guarda el modelo y tokenizador en models/transformer/."""
        if not self._fitted:
            raise RuntimeError("Modelo no entrenado.")
        import torch
        from src.config import MODELS_DIR

        out_dir = Path(directory) if directory else MODELS_DIR / TRANSFORMER_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        self._model.cpu()
        self._model.save_pretrained(str(out_dir))
        self._tokenizer.save_pretrained(str(out_dir))
        self._model.to(self._device)
        logger.info(f"TransformerFraudClassifier guardado en {out_dir}")
        return out_dir

    def load(self, directory: Optional[Path] = None) -> None:
        """Carga el modelo fine-tuneado desde models/transformer/."""
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from src.config import MODELS_DIR

        in_dir = Path(directory) if directory else MODELS_DIR / TRANSFORMER_DIR
        if not in_dir.exists():
            raise FileNotFoundError(f"TransformerFraudClassifier no encontrado: {in_dir}")

        device = self._get_device()
        self._device    = device
        self._tokenizer = AutoTokenizer.from_pretrained(str(in_dir))
        self._model     = AutoModelForSequenceClassification.from_pretrained(
            str(in_dir)
        ).to(device)
        self._model.eval()
        self._fitted = True
        logger.info(f"TransformerFraudClassifier cargado desde {in_dir}")
