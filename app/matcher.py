# app/matcher.py

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def normalise(v: np.ndarray) -> np.ndarray:
    """L2-normalise rows for cosine similarity via dot product."""
    if v.ndim == 1:
        v = v.reshape(1, -1)
    denom = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / denom


def make_doc_text(summary: str, title: str, subtitle: str, description: str, keywords: List[str]) -> str:
    """Construct a single document text used for embedding."""
    kw = ", ".join([str(k).strip() for k in keywords if k and str(k).strip()])
    parts = [
        f"Title: {title}".strip(),
        f"Subtitle: {subtitle}".strip(),
        f"Description: {description}".strip(),
        f"Keywords: {kw}".strip(),
        f"Summary: {summary}".strip(),
    ]
    return "\n".join([p for p in parts if p and not p.endswith(":")])


def item_text(x: Dict[str, Any]) -> str:
    """Text representation for a single vocab item."""
    name = str(x.get("name") or "").strip()
    desc = str(x.get("description") or "").strip()
    if desc:
        return f"{name}\n{desc}".strip()
    return name


def score(doc_vec: np.ndarray, cand_vecs: np.ndarray) -> np.ndarray:
    """Cosine similarity since vectors are L2-normalised."""
    if doc_vec.ndim != 2 or doc_vec.shape[0] != 1:
        raise ValueError(f"doc_vec must be shape (1, d); got {doc_vec.shape}")
    if cand_vecs.ndim != 2:
        raise ValueError(f"cand_vecs must be shape (n, d); got {cand_vecs.shape}")
    if cand_vecs.shape[1] != doc_vec.shape[1]:
        raise ValueError(f"Dim mismatch: cand_vecs {cand_vecs.shape} vs doc_vec {doc_vec.shape}")
    return (cand_vecs @ doc_vec.T).reshape(-1)


def rank(items: List[Dict[str, Any]], scores: np.ndarray) -> List[Tuple[Dict[str, Any], float]]:
    idx = np.argsort(-scores)
    return [(items[i], float(scores[i])) for i in idx]


def adaptive_pick(
    ranked: List[Tuple[Dict[str, Any], float]],
    *,
    min_n: int,
    max_n: int,
    abs_threshold: float,
    rel_to_top: float,
    gap_threshold: float,
) -> List[Tuple[Dict[str, Any], float]]:
    """Same logic as your _adaptive_pick()."""
    if not ranked:
        return []

    top_score = ranked[0][1]
    picked: List[Tuple[Dict[str, Any], float]] = []
    prev_score: float | None = None

    for it, sc in ranked:
        if len(picked) >= max_n:
            break

        if prev_score is not None and len(picked) >= min_n:
            if (prev_score - sc) >= gap_threshold:
                break

        if len(picked) < min_n:
            picked.append((it, sc))
        else:
            if sc >= abs_threshold and sc >= (top_score - rel_to_top):
                picked.append((it, sc))
            else:
                break

        prev_score = sc

    if len(picked) < min_n:
        picked = ranked[:min_n]

    return picked
