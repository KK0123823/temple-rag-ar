# backend/query_chroma.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path
import os

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# ✅ 只有三大神明才做 entity 篩選
DEITY_ENTITIES = {"媽祖", "文昌帝君", "財神爺"}


def _where_eq(key: str, value: str) -> Dict[str, Any]:
    return {key: {"$eq": value}}


def _build_where(entity: Optional[str], section: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    ✅ 關鍵修正：
    - entity 只有在「媽祖/文昌帝君/財神爺」才會生效
    - 廟公/預設/其他分類 → entity 不做過濾（避免把所有資料篩掉）
    """
    conds: List[Dict[str, Any]] = []

    # ⭐ 修正：只允許神明 entity 過濾
    if entity and (entity in DEITY_ENTITIES):
        conds.append(_where_eq("entity", entity))

    # section 若你有用（例如：建築/禁忌/歷史），仍然允許
    if section:
        conds.append(_where_eq("section", section))

    if not conds:
        return None
    if len(conds) == 1:
        return conds[0]
    return {"$and": conds}


# ✅ backend/query_chroma.py 位於 <root>/backend/query_chroma.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CHROMA_DIR = str(PROJECT_ROOT / "data" / "chroma")
DEFAULT_COLLECTION = os.getenv("CHROMA_COLLECTION", "temple_kb")

# ✅ 必須跟 build_vectors_chroma.py 一致
DEFAULT_EMBED_MODEL = os.getenv(
    "CHROMA_EMBED_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

# ✅ 避免每次 query 都重新載入模型（速度差很多）
_MODEL_CACHE: Dict[str, SentenceTransformer] = {}


def _get_model(embed_model_name: str) -> SentenceTransformer:
    m = _MODEL_CACHE.get(embed_model_name)
    if m is None:
        m = SentenceTransformer(embed_model_name)
        _MODEL_CACHE[embed_model_name] = m
    return m


def query(
    query_text: str,
    top_k: int = 4,
    entity: Optional[str] = None,
    section: Optional[str] = None,
    chroma_dir: str = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    embed_model_name: str = DEFAULT_EMBED_MODEL,
) -> List[Dict[str, Any]]:
    """
    回傳：
    [
      {"content": "...", "metadata": {...}, "distance": 0.123},
      ...
    ]

    ✅ 自己算 query embedding，避免 collection embedding_function 跟建庫不一致。
    ✅ 修正 entity 過濾：只對三大神明生效。
    ✅ 加上模型 cache：避免每次 query 重載模型導致超慢。
    """
    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    col = client.get_or_create_collection(name=collection_name)

    where = _build_where(entity, section)

    model = _get_model(embed_model_name)
    q_emb = model.encode([query_text]).tolist()

    res = col.query(
        query_embeddings=q_emb,
        n_results=int(top_k),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    out: List[Dict[str, Any]] = []
    for doc, meta, dist in zip(docs, metas, dists):
        out.append(
            {
                "content": str(doc or "").strip(),
                "metadata": meta or {},
                "distance": float(dist) if dist is not None else 0.0,
            }
        )
    return out


def rag_status(
    chroma_dir: str = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION,
) -> Dict[str, Any]:
    """用來檢查向量庫是否真的有資料（count>0）。"""
    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    col = client.get_or_create_collection(name=collection_name)
    return {
        "chroma_dir": chroma_dir,
        "collection": collection_name,
        "count": int(col.count()),
        "embed_model": DEFAULT_EMBED_MODEL,
    }
