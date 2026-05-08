# backend/api.py
from __future__ import annotations

import os
import re
from typing import Optional, AsyncGenerator, List, Dict, Any, Tuple

import edge_tts
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from chromadb import PersistentClient
from chromadb.config import Settings

from backend.query_chroma import query as query_chroma
from backend.llm_ollama import (
    build_ctx_numbered,
    generate_kids_and_detail,
    sanitize_citations_in_text,
)
from .stt_router import router as stt_router

app = FastAPI(title="Lecheng-Temple-API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(stt_router)

# distance 只做 debug 顯示；不再用固定門檻濾掉資料
RAG_MAX_DISTANCE = float(os.getenv("RAG_MAX_DISTANCE", "0.65"))

VOICE_MAPPING = {
    "mazu": "zh-TW-HsiaoChenNeural",
    "wenchang": "zh-TW-YunJheNeural",
    "caishen": "zh-CN-YunxiNeural",
    "default": "zh-TW-YunJheNeural",
}

# ✅ 只有三大神明才做 entity 篩選；廟公/預設一律全庫搜
DEITY_ENTITIES = {"媽祖", "文昌帝君", "財神爺"}


class AskRequest(BaseModel):
    query: str
    top_k: int = 6
    entity: Optional[str] = "廟公"  # "廟公"/"媽祖"/"文昌帝君"/"財神爺"


def _get_chroma_defaults() -> Tuple[str, str]:
    """確保 /rag_status、/rag_find、keyword candidates 都用同一個庫。"""
    try:
        from backend.query_chroma import DEFAULT_CHROMA_DIR, DEFAULT_COLLECTION
        return DEFAULT_CHROMA_DIR, DEFAULT_COLLECTION
    except Exception:
        chroma_dir = os.getenv("CHROMA_DIR", "data/chroma")
        collection = os.getenv("CHROMA_COLLECTION", "temple_kb")
        return chroma_dir, collection


def _get_collection():
    chroma_dir, collection = _get_chroma_defaults()
    client = PersistentClient(path=chroma_dir, settings=Settings(anonymized_telemetry=False))
    col = client.get_or_create_collection(name=collection)
    return chroma_dir, collection, col


def _filter_contexts(contexts: List[Dict[str, Any]], max_dist: float) -> List[Dict[str, Any]]:
    """
    ✅ 不再用固定 distance threshold 過濾（中文 embedding distance 分佈不穩）。
    只丟掉空內容即可。
    """
    if not contexts:
        return []
    return [c for c in contexts if str(c.get("content", "")).strip()]


def _extract_keywords_zh(text: str) -> List[str]:
    stop = {"什麼", "為什麼", "怎麼", "可以", "請問", "是不是", "一個", "這個", "那個", "廟裡", "廟宇", "地方"}
    kws = re.findall(r"[\u4e00-\u9fff]{2,6}", text)
    out: List[str] = []
    for k in kws:
        if k in stop:
            continue
        if k not in out:
            out.append(k)
    return out[:6]


def _keyword_candidates(query: str, limit: int = 6) -> List[Dict[str, Any]]:
    """
    ✅ Hybrid keyword candidates（不寫死檔名）：
    從 documents / metadata 找包含關鍵字的 chunks，加入候選集合
    """
    _, _, col = _get_collection()
    got = col.get(include=["documents", "metadatas"], limit=5000)

    docs = got.get("documents", []) or []
    metas = got.get("metadatas", []) or []

    keywords = _extract_keywords_zh(query)
    if not keywords:
        return []

    hits: List[Dict[str, Any]] = []
    for doc, meta in zip(docs, metas):
        doc = (doc or "")
        meta = meta or {}
        fn = str(meta.get("file_name", ""))
        src = str(meta.get("source", ""))

        if any(k in doc for k in keywords) or any(k in fn for k in keywords) or any(k in src for k in keywords):
            hits.append({"content": doc.strip(), "metadata": meta, "distance": 0.0})

        if len(hits) >= limit:
            break

    return hits


def _merge_contexts(keyword_ctx: List[Dict[str, Any]], vector_ctx: List[Dict[str, Any]], final_limit: int) -> List[Dict[str, Any]]:
    """
    合併 keyword + vector 結果並去重：
    - keyword 放前面（提升精準名詞命中率）
    - 用 (file_name, section, content前50字) 去重
    """
    def _key(c: Dict[str, Any]) -> Tuple[str, str, str]:
        m = c.get("metadata", {}) or {}
        return (
            str(m.get("file_name", "")),
            str(m.get("section", "")),
            str((c.get("content", "") or "")[:50]),
        )

    seen = set()
    merged: List[Dict[str, Any]] = []

    for c in (keyword_ctx or []) + (vector_ctx or []):
        if not str(c.get("content", "")).strip():
            continue
        k = _key(c)
        if k in seen:
            continue
        seen.add(k)
        merged.append(c)

    return merged[:final_limit]


@app.get("/rag_status")
def rag_status():
    chroma_dir, collection, col = _get_collection()
    return {
        "chroma_dir": chroma_dir,
        "collection": collection,
        "count": int(col.count()),
        "rag_max_distance": RAG_MAX_DISTANCE,
    }


@app.get("/rag_find")
def rag_find(q: str = Query(..., description="用關鍵字找 documents/metadata 是否包含該字（確認資料是否在庫內）")):
    chroma_dir, collection, col = _get_collection()

    got = col.get(include=["documents", "metadatas"], limit=2000)
    docs = got.get("documents", []) or []
    metas = got.get("metadatas", []) or []

    hits = []
    for doc, meta in zip(docs, metas):
        doc = doc or ""
        meta = meta or {}
        fn = meta.get("file_name", "")
        src = meta.get("source", "")
        if (q in doc) or (q in fn) or (q in src):
            hits.append({
                "file_name": fn,
                "source": src,
                "entity": meta.get("entity", ""),
                "section": meta.get("section", ""),
                "snippet": doc[:140],
            })
        if len(hits) >= 10:
            break

    return {"q": q, "hit_count": len(hits), "hits": hits, "chroma_dir": chroma_dir, "collection": collection}


@app.post("/ask")
def ask(req: AskRequest):
    mapping = {"媽祖": "mazu", "文昌帝君": "wenchang", "財神爺": "caishen", "廟公": "default", "default": "default"}
    entity_in = (req.entity or "廟公").strip()
    char_key = mapping.get(entity_in, "default")

    # ✅ 只有三大神明才做 entity 篩選；廟公/預設一律全庫搜
    search_ent = entity_in if entity_in in DEITY_ENTITIES else None

    # 1) 向量查詢
    raw_contexts = query_chroma(query_text=req.query, top_k=req.top_k, entity=search_ent)
    raw_matches = len(raw_contexts)
    vector_ctx = _filter_contexts(raw_contexts, RAG_MAX_DISTANCE)

    entity_fallback = False
    if (not vector_ctx) and search_ent:
        raw2 = query_chroma(query_text=req.query, top_k=req.top_k, entity=None)
        vector_ctx2 = _filter_contexts(raw2, RAG_MAX_DISTANCE)
        if vector_ctx2:
            vector_ctx = vector_ctx2
            entity_fallback = True

    # 2) ✅ Hybrid：加 keyword candidates 提升精準名詞（例如天井）命中
    kw_limit = max(4, req.top_k // 2)
    keyword_ctx = _keyword_candidates(req.query, limit=kw_limit)

    final_limit = max(req.top_k, 8)
    contexts = _merge_contexts(keyword_ctx, vector_ctx, final_limit=final_limit)

    # ✅ 關鍵：只餵前 4 段給 LLM，避免回答爆長
    contexts_for_llm = contexts[:4]

    ctx_text, cites = ("", 0)
    used_files: List[str] = []

    if contexts_for_llm:
        ctx_text, citations = build_ctx_numbered(contexts_for_llm)
        cites = len(citations)
        for c in contexts_for_llm:
            fn = (c.get("metadata", {}) or {}).get("file_name", "")
            if fn and fn not in used_files:
                used_files.append(fn)

    kids, panel = generate_kids_and_detail(query=req.query, ctx_text=ctx_text, persona_key=char_key)

    kids_clean = sanitize_citations_in_text((kids or "").strip(), max_n=cites)
    panel_clean = sanitize_citations_in_text((panel or "").strip(), max_n=cites) if panel else ""

    top_distance = None
    if vector_ctx:
        try:
            top_distance = float(vector_ctx[0].get("distance", None))
        except Exception:
            top_distance = None

    return {
        "ok": True,
        "kids": kids_clean,          # ✅ 給 TTS 的短語音
        "panel": panel_clean,        # ✅ 給 UI 面板的較完整內容
        "character": char_key,
        "rag_used": bool(contexts),
        "matches": len(contexts),
        "top_distance": top_distance,
        "rag_max_distance": RAG_MAX_DISTANCE,
        "entity_fallback": entity_fallback,
        "raw_matches": raw_matches,
        "kw_matches": len(keyword_ctx),
        "used_files": used_files,
    }


async def stream_tts_gen(text: str, voice: str) -> AsyncGenerator[bytes, None]:
    communicate = edge_tts.Communicate(text, voice)
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            yield chunk["data"]


@app.get("/tts")
async def text_to_speech(text: str, deity: str = "default"):
    # ✅ 保險：TTS 一律短（避免偶發爆字）
    clean = re.sub(r"\[\d+\]", "", text).replace("～", "，").strip()
    clean = clean.replace("\n", " ")
    clean = re.sub(r"\s+", " ", clean).strip()
    clean = clean[:140]  # 硬上限：最多 140 字元（可自行調整）

    voice = VOICE_MAPPING.get(deity, VOICE_MAPPING["default"])
    return StreamingResponse(stream_tts_gen(clean, voice), media_type="audio/mpeg")
