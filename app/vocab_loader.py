# app/vocab_loader.py

from __future__ import annotations

import json

from pathlib import Path
from typing import Any, Dict, List


def load_vocab_from_dir(vocab_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Loads controlled vocabs from data_model_v2/*.json

    Output matches your Django-side structure:
    - each item has {id, name, description?, uri?, parent_category_id?}
    - only Published items
    """
    base = Path(vocab_dir)

    def load_one(filename: str) -> list[dict]:
        p = base / filename
        items = json.loads(p.read_text(encoding="utf-8"))

        out: list[dict] = []
        for it in items:
            if str(it.get("status", "")).lower() != "published":
                continue

            _id = str(it.get("_id") or "").strip()
            name = str(it.get("name") or "").strip()
            if not _id or not name:
                continue

            row: dict[str, Any] = {"id": _id, "name": name}

            # Keep parent relations for subcategories
            if filename.endswith("subcategories.json"):
                pci = it.get("parent_category_id") or []
                if isinstance(pci, list):
                    row["parent_category_id"] = [str(x).strip() for x in pci if str(x).strip()]
                else:
                    row["parent_category_id"] = [str(pci).strip()] if str(pci).strip() else []

            desc = it.get("description") or ""
            if isinstance(desc, str) and desc.strip():
                row["description"] = desc.strip()

            uri = it.get("uri") or ""
            if isinstance(uri, str) and uri.strip():
                row["uri"] = uri.strip()

            out.append(row)

        return out

    return {
        "category": load_one("data_model.category.json"),
        "intended_purposes": load_one("data_model.intended_purposes.json"),
        "subcategories": load_one("data_model.subcategories.json"),
        "topics": load_one("data_model.topics.json"),
        "themes": load_one("data_model.themes.json"),
    }
