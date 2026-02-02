# app/main.py

from __future__ import annotations

import logging

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .config import settings
from .matcher import adaptive_pick, make_doc_text, rank, score
from .state import ServiceState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STATE = ServiceState(
    model_name=settings.MODEL_NAME,
    vocab_dir=settings.VOCAB_DIR,
    embed_batch_size=settings.EMBED_BATCH_SIZE,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model + vocab once at startup; keep in memory for the lifetime of the process.
    """
    STATE.load_all()
    yield

app = FastAPI(title="vocab-selector", version="1.0.0", lifespan=lifespan)


class DocIn(BaseModel):
    summary: str = Field(default="")
    title: str = Field(default="")
    subtitle: str = Field(default="")
    description: str = Field(default="")
    keywords: List[str] = Field(default_factory=list)


class OptionsIn(BaseModel):
    max_list: int = 8
    abs_threshold: float = 0.30
    rel_to_top: float = 0.08
    gap_threshold: float = 0.07
    include_scores: bool = False


class SelectRequest(BaseModel):
    doc: DocIn
    options: OptionsIn = Field(default_factory=OptionsIn)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "model_loaded": STATE.model is not None,
        "vocab_kinds": sorted(list(STATE.vocab.keys())),
    }


@app.post("/v1/select-vocab")
def select_vocab(req: SelectRequest) -> Dict[str, Any]:
    """
    Returns:
    - category: str (top-1)
    - topics/themes/intended_purposes/subcategories: list[str] (adaptive pick)
    Optional: *_scored if include_scores=True
    """
    doc = req.doc
    opt = req.options

    # Build doc text exactly as in Django matcher
    doc_text = make_doc_text(doc.summary, doc.title, doc.subtitle, doc.description, doc.keywords)

    doc_vec = STATE.embed_texts([doc_text])  # shape (1, d)
    out: Dict[str, Any] = {}

    # Category (top-1)
    cats = STATE.vocab.get("category", [])
    if not cats:
        out["category"] = ""
        if opt.include_scores:
            out["category_scored"] = []
        chosen_cat_id = ""
    else:
        cat_vecs = STATE.get_vocab_embeds("category", cats)
        cat_scores = score(doc_vec, cat_vecs)
        ranked = rank(cats, cat_scores)
        chosen_cat_item = ranked[0][0]
        chosen_cat_id = str(chosen_cat_item.get("id") or "").strip()
        out["category"] = str(chosen_cat_item.get("name") or "")
        if opt.include_scores:
            out["category_scored"] = [(out["category"], ranked[0][1])]

    def pick_many(kind: str, min_n: int, items_override: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        items = items_override if items_override is not None else STATE.vocab.get(kind, [])
        if not items:
            if opt.include_scores:
                out[f"{kind}_scored"] = []
            return []

        vecs = STATE._ensure_vocab_embeds(kind, items)
        scores_arr = score(doc_vec, vecs)
        ranked_items = rank(items, scores_arr)

        picked = adaptive_pick(
            ranked_items,
            min_n=min_n,
            max_n=min(opt.max_list, len(ranked_items)),
            abs_threshold=opt.abs_threshold,
            rel_to_top=opt.rel_to_top,
            gap_threshold=opt.gap_threshold,
        )

        names: List[str] = []
        for it, _sc in picked:
            n = str(it.get("name") or "").strip()
            if n and n not in names:
                names.append(n)

        if opt.include_scores:
            out[f"{kind}_scored"] = [(str(it.get("name") or ""), sc) for it, sc in picked]

        return names

    out["topics"] = pick_many("topics", min_n=1)
    out["themes"] = pick_many("themes", min_n=1)
    out["intended_purposes"] = pick_many("intended_purposes", min_n=1)

    # Subcategories filtered by chosen category
    subcats_all = STATE.vocab.get("subcategories", [])
    if chosen_cat_id:
        subcats_filtered: List[Dict[str, Any]] = []
        for sc in subcats_all:
            parent_ids = sc.get("parent_category_id") or []
            if isinstance(parent_ids, list) and chosen_cat_id in [str(x).strip() for x in parent_ids]:
                subcats_filtered.append(sc)
    else:
        subcats_filtered = subcats_all

    if not subcats_filtered:
        subcats_filtered = subcats_all

    out["subcategories"] = pick_many("subcategories", min_n=1, items_override=subcats_filtered)
    return out


@app.post("/v1/reload")
def reload_vocab(x_reload_token: str = Header(default="")) -> Dict[str, Any]:
    """
    Optional admin endpoint for reloading vocab without redeploy.
    Protect with a token.
    """
    if settings.RELOAD_TOKEN:
        if not x_reload_token or x_reload_token != settings.RELOAD_TOKEN:
            raise HTTPException(status_code=401, detail="unauthorised")
    STATE.reload_vocab()
    return {"ok": True, "vocab_kinds": sorted(list(STATE.vocab.keys()))}
