"""
Módulo de embeddings semánticos con Mistral AI.
Construye un índice de referencia con mensajes conocidos y permite
buscar los más similares a un mensaje nuevo por similitud coseno.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except ImportError:
    pass

INDEX_FILE = "semantic_index.npz"


class SemanticIndex:
    """
    Índice semántico de mensajes de referencia.

    Flujo de uso:
        1. build(df)      — embide una muestra del dataset y crea el índice
        2. save()         — persiste en models/semantic_index.npz
        3. load()         — carga el índice en memoria (arranque del predictor)
        4. search(text)   — devuelve similitudes y vecinos más cercanos
    """

    EMBED_MODEL = "mistral-embed"
    BATCH_SIZE  = 64

    def __init__(self, api_key: Optional[str] = None, model: str = EMBED_MODEL) -> None:
        key = api_key or os.environ.get("MISTRAL_API_KEY", "")
        if not key:
            raise ValueError("MISTRAL_API_KEY no encontrada.")
        try:
            from mistralai.client.sdk import Mistral
            self.client = Mistral(api_key=key)
        except ImportError as exc:
            raise ImportError("pip install mistralai") from exc
        self.model    = model
        self.vectors:  Optional[np.ndarray] = None
        self.labels:   Optional[np.ndarray] = None
        self.messages: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Construcción
    # ------------------------------------------------------------------

    def build(self, df, sample_per_class: int = 500) -> None:
        import pandas as pd

        frames = []
        for label, group in df.groupby("label"):
            n = min(len(group), sample_per_class)
            frames.append(group.sample(n, random_state=42))
        sample = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=42)

        texts = sample["message"].astype(str).tolist()
        dist  = sample["label"].value_counts().to_dict()
        logger.info(f"Embidiendo {len(texts)} mensajes — distribución: {dist}")

        vecs = self._embed(texts)
        self.vectors  = vecs
        self.labels   = np.array(sample["label"].values, dtype=object)
        self.messages = np.array([t[:200] for t in texts], dtype=object)
        logger.info(f"Índice construido: {len(texts)} vectores de {vecs.shape[1]} dims.")

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> Path:
        if path is None:
            from src.config import MODELS_DIR
            path = MODELS_DIR / INDEX_FILE
        np.savez_compressed(
            str(path),
            vectors=self.vectors,
            labels=self.labels.astype(str),
            messages=self.messages.astype(str),
        )
        logger.info(f"Índice guardado en {path}")
        return Path(path)

    def load(self, path: Optional[Path] = None) -> None:
        if path is None:
            from src.config import MODELS_DIR
            path = MODELS_DIR / INDEX_FILE
        if not Path(path).exists():
            raise FileNotFoundError(f"Índice semántico no encontrado: {path}")
        data = np.load(str(path), allow_pickle=True)
        self.vectors  = data["vectors"].astype(np.float32)
        self.labels   = data["labels"]
        self.messages = data["messages"]
        logger.info(f"Índice cargado: {len(self.labels)} vectores desde {path}")

    # ------------------------------------------------------------------
    # Búsqueda
    # ------------------------------------------------------------------

    def search(self, text: str, top_k: int = 5) -> dict:
        if self.vectors is None:
            raise RuntimeError("Índice no cargado. Llama a load() o build() primero.")

        from sklearn.metrics.pairwise import cosine_similarity

        query_vec  = self._embed([text])
        sims       = cosine_similarity(query_vec, self.vectors)[0]
        fraud_mask = self.labels == "fraudulent"
        legit_mask = self.labels == "legitimate"

        fraud_sim = float(sims[fraud_mask].max()) if fraud_mask.any() else 0.0
        legit_sim = float(sims[legit_mask].max()) if legit_mask.any() else 0.0

        top_idx   = np.argsort(sims)[::-1][:top_k]
        neighbors = [
            {
                "message":    str(self.messages[i]),
                "label":      str(self.labels[i]),
                "similarity": round(float(sims[i]), 4),
            }
            for i in top_idx
        ]
        return {
            "fraud_similarity": round(fraud_sim, 4),
            "legit_similarity": round(legit_sim, 4),
            "top_neighbors":    neighbors,
        }

    # ------------------------------------------------------------------
    # Embedding interno
    # ------------------------------------------------------------------

    def _embed(self, texts: list[str]) -> np.ndarray:
        all_vecs = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            resp  = self.client.embeddings.create(model=self.model, inputs=batch)
            all_vecs.extend([item.embedding for item in resp.data])
        return np.array(all_vecs, dtype=np.float32)
