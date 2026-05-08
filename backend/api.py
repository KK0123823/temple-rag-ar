from __future__ import annotations

import os
import re
from typing import Optional, List, Dict, Any, Tuple

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chromadb import PersistentClient
from chromadb.config import Settings

from backend.query_chroma import query as query_chroma


app = FastAPI(title="Temple RAG API for AR")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RAG_MAX_DISTANCE = float(os.getenv("RAG_MAX_DISTANCE", "0.65"))
DEITY_ENTITIES = {"媽祖", "文昌帝君", "財神爺"}


class AskRequest(BaseModel):
    query: str
    top_k: int = 6
    entity: Optional[str] = "廟公"


def _get_chroma_defaults() -> Tuple[str, str]:
    try:
        from backend.query_chroma import DEFAULT_CHROMA_DIR, DEFAULT_COLLECTION
        return DEFAULT_CHROMA_DIR, DEFAULT_COLLECTION
    except Exception:
        chroma_dir = os.getenv("CHROMA_DIR", "data/chroma")
        collection = os.getenv("CHROMA_COLLECTION", "temple_kb")
        return chroma_dir, collection


def _get_collection():
    chroma_dir, collection = _get_chroma_defaults()
    client = PersistentClient(
        path=chroma_dir,
        settings=Settings(anonymized_telemetry=False)
    )
    col = client.get_or_create_collection(name=collection)
    return chroma_dir, collection, col


def _extract_keywords_zh(text: str) -> List[str]:
    stop = {
        "什麼", "為什麼", "怎麼", "可以", "請問", "是不是",
        "一個", "這個", "那個", "廟裡", "廟宇", "地方"
    }

    kws = re.findall(r"[\u4e00-\u9fff]{2,6}", text)
    out = []

    for k in kws:
        if k in stop:
            continue
        if k not in out:
            out.append(k)

    return out[:6]


def _keyword_candidates(query: str, limit: int = 6) -> List[Dict[str, Any]]:
    _, _, col = _get_collection()
    got = col.get(include=["documents", "metadatas"], limit=5000)

    docs = got.get("documents", []) or []
    metas = got.get("metadatas", []) or []

    keywords = _extract_keywords_zh(query)
    if not keywords:
        return []

    hits = []

    for doc, meta in zip(docs, metas):
        doc = doc or ""
        meta = meta or {}

        file_name = str(meta.get("file_name", ""))
        source = str(meta.get("source", ""))

        if (
            any(k in doc for k in keywords)
            or any(k in file_name for k in keywords)
            or any(k in source for k in keywords)
        ):
            hits.append({
                "content": doc.strip(),
                "metadata": meta,
                "distance": 0.0
            })

        if len(hits) >= limit:
            break

    return hits


def _filter_contexts(contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not contexts:
        return []

    return [
        c for c in contexts
        if str(c.get("content", "")).strip()
    ]


def _merge_contexts(
    keyword_ctx: List[Dict[str, Any]],
    vector_ctx: List[Dict[str, Any]],
    final_limit: int
) -> List[Dict[str, Any]]:

    def _key(c: Dict[str, Any]):
        meta = c.get("metadata", {}) or {}
        return (
            str(meta.get("file_name", "")),
            str(meta.get("section", "")),
            str((c.get("content", "") or "")[:50]),
        )

    merged = []
    seen = set()

    for c in (keyword_ctx or []) + (vector_ctx or []):
        if not str(c.get("content", "")).strip():
            continue

        k = _key(c)
        if k in seen:
            continue

        seen.add(k)
        merged.append(c)

    return merged[:final_limit]


def _build_answer(query: str, contexts: List[Dict[str, Any]], entity: str) -> Tuple[str, str]:
    """
    Render 版：不呼叫本機 Ollama。
    先用 RAG 檢索內容組合回答，確保 AR 可以穩定使用。
    """

    if not contexts:
        kids = "我目前沒有在資料庫中找到明確資料，可以再換個方式問我喔。"
        panel = f"問題：{query}\n\n目前 RAG 沒有找到相關資料。"
        return kids, panel

    snippets = []

    for i, c in enumerate(contexts[:3], start=1):
        content = str(c.get("content", "")).strip()
        meta = c.get("metadata", {}) or {}

        file_name = meta.get("file_name", "")
        section = meta.get("section", "")
        source = file_name or source if (source := meta.get("source", "")) else ""

        short = content[:180]
        snippets.append({
            "index": i,
            "content": short,
            "source": source,
            "section": section
        })

    first = snippets[0]["content"]

    if entity in {"媽祖", "文昌帝君", "財神爺"}:
        kids = f"我是{entity}。根據資料，{first}"
    else:
        kids = f"根據廟宇文化資料，{first}"

    kids = kids[:180]

    panel_lines = [
        f"問題：{query}",
        f"角色：{entity}",
        "",
        "RAG 檢索結果："
    ]

    for s in snippets:
        panel_lines.append("")
        panel_lines.append(f"[{s['index']}] {s['content']}")
        if s["source"]:
            panel_lines.append(f"來源：{s['source']}")
        if s["section"]:
            panel_lines.append(f"段落：{s['section']}")

    panel = "\n".join(panel_lines)

    return kids, panel


@app.get("/")
def root():
    return {
        "ok": True,
        "message": "Temple RAG API is running on Render.",
        "docs": "/docs",
        "ask": "/ask",
        "rag_status": "/rag_status"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.get("/rag_status")
def rag_status():
    chroma_dir, collection, col = _get_collection()

    return {
        "ok": True,
        "chroma_dir": chroma_dir,
        "collection": collection,
        "count": int(col.count()),
        "rag_max_distance": RAG_MAX_DISTANCE,
    }


@app.get("/rag_find")
def rag_find(
    q: str = Query(..., description="用關鍵字找 documents/metadata 是否包含該字")
):
    chroma_dir, collection, col = _get_collection()

    got = col.get(include=["documents", "metadatas"], limit=2000)
    docs = got.get("documents", []) or []
    metas = got.get("metadatas", []) or []

    hits = []

    for doc, meta in zip(docs, metas):
        doc = doc or ""
        meta = meta or {}

        file_name = meta.get("file_name", "")
        source = meta.get("source", "")

        if (q in doc) or (q in file_name) or (q in source):
            hits.append({
                "file_name": file_name,
                "source": source,
                "entity": meta.get("entity", ""),
                "section": meta.get("section", ""),
                "snippet": doc[:160],
            })

        if len(hits) >= 10:
            break

    return {
        "ok": True,
        "q": q,
        "hit_count": len(hits),
        "hits": hits,
        "chroma_dir": chroma_dir,
        "collection": collection
    }


@app.post("/ask")
def ask(req: AskRequest):
    entity_in = (req.entity or "廟公").strip()

    raw_contexts = []
    vector_ctx = []
    entity_fallback = False

    keyword_ctx = _keyword_candidates(
        req.query,
        limit=max(6, req.top_k)
    )

    contexts = keyword_ctx[:max(req.top_k, 8)]

    kids, panel = _build_answer(
        query=req.query,
        contexts=contexts,
        entity=entity_in
    )

    used_files = []

    for c in contexts[:4]:
        meta = c.get("metadata", {}) or {}
        file_name = meta.get("file_name", "") or meta.get("source", "")
        if file_name and file_name not in used_files:
            used_files.append(file_name)

    return {
        "ok": True,
        "kids": kids,
        "panel": panel,
        "character": entity_in,
        "rag_used": bool(contexts),
        "matches": len(contexts),
        "top_distance": None,
        "rag_max_distance": RAG_MAX_DISTANCE,
        "entity_fallback": entity_fallback,
        "raw_matches": len(raw_contexts),
        "kw_matches": len(keyword_ctx),
        "used_files": used_files,
    }