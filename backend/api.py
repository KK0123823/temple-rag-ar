# backend/api.py
from __future__ import annotations

import os
import re
import json
from typing import Optional, AsyncGenerator, List, Dict, Any, Tuple

import edge_tts
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from chromadb import PersistentClient
from chromadb.config import Settings

from backend.query_chroma import query as query_chroma
from backend.llm_ollama import (
    build_ctx_numbered,
    generate_kids_and_detail,
    sanitize_citations_in_text,
)
#from .stt_router import router as stt_router


app = FastAPI(title="Lecheng-Temple-API")

app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#app.include_router(stt_router)

RAG_MAX_DISTANCE = float(os.getenv("RAG_MAX_DISTANCE", "0.65"))

VOICE_MAPPING = {
    "mazu": "zh-TW-HsiaoChenNeural",
    "wenchang": "zh-TW-YunJheNeural",
    "caishen": "zh-CN-YunxiNeural",
    "default": "zh-TW-YunJheNeural",
}

DEITY_ENTITIES = {"媽祖", "文昌帝君", "財神爺"}


class AskRequest(BaseModel):
    query: Optional[str] = None
    question: Optional[str] = None
    top_k: int = 6
    entity: Optional[str] = "廟公"
    section: Optional[str] = None
    use_llm: Optional[bool] = True


def _clean_llm_json_text(text: str, key: str = "kids") -> str:
    if not text:
        return ""

    t = str(text).strip()

    # 移除 markdown code block
    t = t.replace("```json", "").replace("```", "").strip()

    # 嘗試找出第一個 JSON 物件
    start = t.find("{")
    end = t.rfind("}")

    if start != -1 and end != -1 and end > start:
        json_part = t[start:end + 1]
        try:
            obj = json.loads(json_part)
            if isinstance(obj, dict):
                value = obj.get(key, "")
                if value:
                    return str(value).strip()
        except Exception:
            pass

    # 如果 JSON 解析失敗，用正則硬抓 kids/panel
    pattern = rf'"{key}"\s*:\s*"([^"]*)'
    m = re.search(pattern, t)
    if m:
        return m.group(1).strip()

    return t

def _final_clean_text(text: str) -> str:
    if not text:
        return ""

    t = str(text)

    # 移除 ANSI 控制碼
    t = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", t)
    t = t.replace("\x1b", "")
    t = t.replace("-[K", "")
    t = t.replace("[K", "")

    # 清掉 markdown json block
    t = t.replace("```json", "")
    t = t.replace("```", "")

    # 換行與跳脫字元
    t = t.replace("\\n", "\n")
    t = t.replace("\\t", "")
    t = t.replace("\t", "")

    # 清除中文之間多餘空白：例「媽祖 本名」→「媽祖本名」
    t = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", t)

    # 清除標點前後多餘空白
    t = re.sub(r"\s+([，。！？；：、）】》])", r"\1", t)
    t = re.sub(r"([（【《])\s+", r"\1", t)

    # 多個空白合併成一個
    t = re.sub(r"[ ]{2,}", " ", t)

    return t.strip()


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


def _filter_contexts(contexts: List[Dict[str, Any]], max_dist: float) -> List[Dict[str, Any]]:
    if not contexts:
        return []
    return [c for c in contexts if str(c.get("content", "")).strip()]


def _extract_keywords_zh(text: str) -> List[str]:
    stop = {
        "什麼", "為什麼", "怎麼", "可以", "請問", "是不是",
        "一個", "這個", "那個", "廟裡", "廟宇", "地方"
    }

    kws = re.findall(r"[\u4e00-\u9fff]{2,6}", text)
    out: List[str] = []

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

    hits: List[Dict[str, Any]] = []

    for doc, meta in zip(docs, metas):
        doc = doc or ""
        meta = meta or {}

        fn = str(meta.get("file_name", ""))
        src = str(meta.get("source", ""))
        entity = str(meta.get("entity", ""))
        section = str(meta.get("section", ""))

        search_area = f"{doc} {fn} {src} {entity} {section}"

        if any(k in search_area for k in keywords):
            hits.append({
                "content": doc.strip(),
                "metadata": meta,
                "distance": 0.0
            })

        if len(hits) >= limit:
            break

    return hits


def _merge_contexts(
    keyword_ctx: List[Dict[str, Any]],
    vector_ctx: List[Dict[str, Any]],
    final_limit: int
) -> List[Dict[str, Any]]:

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


@app.get("/")
def root():
    return {
        "ok": True,
        "message": "Lecheng Temple API is running.",
        "frontend": "/static/index.html",
        "docs": "/docs",
        "rag_status": "/rag_status",
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
def rag_find(q: str = Query(..., description="用關鍵字找 documents/metadata 是否包含該字")):
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
    query_text = (req.query or req.question or "").strip()

    if not query_text:
        return {
            "ok": False,
            "error": "query is required",
            "kids": "請先輸入問題喔。",
            "detail": "缺少 query 或 question 欄位。",
            "panel": "缺少 query 或 question 欄位。",
            "citations": [],
            "character": "default",
        }

    mapping = {
        "媽祖": "mazu",
        "文昌帝君": "wenchang",
        "財神爺": "caishen",
        "廟公": "default",
        "default": "default"
    }

    entity_in = (req.entity or "廟公").strip()
    char_key = mapping.get(entity_in, "default")

    mentioned_entities = [
    e for e in DEITY_ENTITIES
    if e in query_text
]

    # 多神明問題 → 不限制 entity
    if len(mentioned_entities) >= 2:
        search_ent = None
    else:
        search_ent = entity_in if entity_in in DEITY_ENTITIES else None

    raw_contexts = query_chroma(
        query_text=query_text,
        top_k=req.top_k,
        entity=search_ent
    )
    raw_matches = len(raw_contexts)

    vector_ctx = _filter_contexts(raw_contexts, RAG_MAX_DISTANCE)

    entity_fallback = False
    if (not vector_ctx) and search_ent:
        raw2 = query_chroma(
            query_text=query_text,
            top_k=req.top_k,
            entity=None
        )
        vector_ctx2 = _filter_contexts(raw2, RAG_MAX_DISTANCE)

        if vector_ctx2:
            vector_ctx = vector_ctx2
            entity_fallback = True

    kw_limit = max(4, req.top_k // 2)
    keyword_ctx = _keyword_candidates(query_text, limit=kw_limit)

    final_limit = max(req.top_k, 8)
    contexts = _merge_contexts(
        keyword_ctx=keyword_ctx,
        vector_ctx=vector_ctx,
        final_limit=final_limit
    )

    contexts_for_llm = contexts[:4]

    ctx_text = ""
    cites = 0
    used_files: List[str] = []

    if contexts_for_llm:
        ctx_text, citations = build_ctx_numbered(contexts_for_llm)
        cites = len(citations)

        for c in contexts_for_llm:
            fn = (c.get("metadata", {}) or {}).get("file_name", "")
            if fn and fn not in used_files:
                used_files.append(fn)

    if req.use_llm:
        kids, panel = generate_kids_and_detail(
            query=query_text,
            ctx_text=ctx_text,
            persona_key=char_key
        )
    else:
        kids = contexts_for_llm[0]["content"][:180] if contexts_for_llm else "目前沒有找到相關資料。"
        panel = "\n\n".join([c["content"] for c in contexts_for_llm]) if contexts_for_llm else "目前沒有找到相關資料。"

    kids_clean = _final_clean_text(
    sanitize_citations_in_text(
        _clean_llm_json_text(kids, "kids"),
        max_n=cites
    )
)

    panel_clean = _final_clean_text(
        sanitize_citations_in_text(
        _clean_llm_json_text(panel or kids, "panel"),
        max_n=cites
    )
)

    top_distance = None

    if vector_ctx:
        try:
            top_distance = float(vector_ctx[0].get("distance", None))
        except Exception:
            top_distance = None

    return {
        "ok": True,

        "kids": kids_clean,
        "detail": panel_clean,
        "panel": panel_clean,
        "citations": used_files,

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
    clean = re.sub(r"\[\d+\]", "", text).replace("～", "，").strip()
    clean = clean.replace("\n", " ")
    clean = re.sub(r"\s+", " ", clean).strip()
    clean = clean[:140]

    voice = VOICE_MAPPING.get(deity, VOICE_MAPPING["default"])

    return StreamingResponse(
        stream_tts_gen(clean, voice),
        media_type="audio/mpeg"
    )