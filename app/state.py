# app/state.py

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from .matcher import item_text, normalise
from .vocab_loader import load_vocab_from_dir


logger = logging.getLogger(__name__)


class ServiceState:
    """
    Holds:
    - SentenceTransformer model (loaded once)
    - vocab dict (loaded once)
    - precomputed embeddings per vocab kind (cached)
    """
    def __init__(self, *, model_name: str, vocab_dir: str, embed_batch_size: int = 32) -> None:
        self.model_name = model_name
        self.vocab_dir = vocab_dir
        self.embed_batch_size = embed_batch_size

        self.model: SentenceTransformer | None = None
        self.vocab: Dict[str, List[Dict[str, Any]]] = {}
        self.vocab_embeds: Dict[str, Tuple[Tuple[str, ...], np.ndarray]] = {}
        # store ids tuple so we know if cached embeds match current vocab

    def get_vocab_embeds(self, kind: str, items: List[Dict[str, Any]]) -> np.ndarray:
        return self._ensure_vocab_embeds(kind, items)

    def load_all(self) -> None:
        logger.info("Loading SentenceTransformer model=%s", self.model_name)
        self.model = SentenceTransformer(self.model_name)

        logger.info("Loading vocab from %s", self.vocab_dir)
        self.vocab = load_vocab_from_dir(self.vocab_dir)

        logger.info("Precomputing vocab embeddings…")
        self.vocab_embeds = {}
        for kind, items in self.vocab.items():
            self._ensure_vocab_embeds(kind, items)
        logger.info("Vocab embeddings ready.")

    def reload_vocab(self) -> None:
        """Reload vocab and recompute embeddings (model stays loaded)."""
        logger.info("Reloading vocab from %s", self.vocab_dir)
        self.vocab = load_vocab_from_dir(self.vocab_dir)
        self.vocab_embeds = {}
        for kind, items in self.vocab.items():
            self._ensure_vocab_embeds(kind, items)
        logger.info("Vocab reload complete.")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed + L2-normalise."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        vecs = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=self.embed_batch_size,
        )
        return normalise(vecs)

    def _ensure_vocab_embeds(self, kind: str, items: List[Dict[str, Any]]) -> np.ndarray:
        """
        Cache embeddings for a vocab list.
        Cache key uses tuple(ids) to detect drift.
        """
        ids = tuple(str(x.get("id") or "") for x in items)
        cached = self.vocab_embeds.get(kind)
        if cached is not None and cached[0] == ids:
            return cached[1]

        vecs = self.embed_texts([item_text(it) for it in items])
        self.vocab_embeds[kind] = (ids, vecs)
        return vecs
