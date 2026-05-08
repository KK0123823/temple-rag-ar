# build_vectors_chroma.py
import json
import os
from pathlib import Path
from typing import Dict, Any, List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent

CHUNKS_PATH = PROJECT_ROOT / "data" / "kb" / "chunks.jsonl"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "temple_kb")

# ✅ 必須跟 query_chroma.py 一致
EMBED_MODEL = os.getenv(
    "CHROMA_EMBED_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

def main():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"找不到 chunks.jsonl: {CHUNKS_PATH}")

    chunks: List[Dict[str, Any]] = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    print(f"[INFO] 讀取到 {len(chunks)} 段文本，開始建立向量庫...")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    model = SentenceTransformer(EMBED_MODEL)

    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []

    for ch in chunks:
        content = (ch.get("content") or "").strip()
        if not content:
            continue
        meta = ch.get("metadata") or {}
        cid = meta.get("chunk_id") or f"chunk_{len(ids)}"
        texts.append(content)
        metadatas.append(meta)
        ids.append(str(cid))

    if not ids:
        raise RuntimeError("chunks.jsonl 內沒有可用內容（texts 為空）。")

    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True).tolist()

    # ✅ 最乾淨：整個 collection 刪掉重建（避免殘留舊設定/舊資料）
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    col = client.get_or_create_collection(name=COLLECTION_NAME)
    col.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)

    print("[OK] 向量庫建立完成")
    print(f" - collection: {COLLECTION_NAME}")
    print(f" - chroma_dir: {CHROMA_DIR}")
    print(f" - count: {col.count()}")
    print(f" - embed_model: {EMBED_MODEL}")

if __name__ == "__main__":
    main()
