# app/config.py

from __future__ import annotations

import os

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    VOCAB_DIR: str = os.environ.get("VOCAB_DIR", "data_model_v2")

    MODEL_NAME: str = os.environ.get("MODEL_NAME", "BAAI/bge-base-en-v1.5")

    RELOAD_TOKEN: str = os.environ.get("RELOAD_TOKEN", "")

    EMBED_BATCH_SIZE: int = int(os.environ.get("EMBED_BATCH_SIZE", "32"))


settings = Settings()
