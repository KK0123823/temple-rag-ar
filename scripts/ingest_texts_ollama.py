# -*- coding: utf-8 -*-
# scripts/ingest_texts_fastembed.py
#
# 將 data/texts/<topic>/*.txt|*.md 讀入、切段、向量化，寫入 Postgres+pgvector
# 與現有的圖片 chunks 共用同一張表：
# - chunks.embedding 固定使用 vector(512)
# - fastembed 文字向量若是 384 維，就在尾端補 0 到 512 維
#
# 資料表：
# documents(doc_id, topic, filename, media_type, language, text_content)
# chunks(chunk_id, doc_id, idx, content, metadata(jsonb:{topic,file,source,index}), embedding vector(512))

import os
import json
import uuid
from pathlib import Path
from typing import List, Iterable, Tuple

import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector

from fastembed import TextEmbedding

# ================== 設定 ==================
# ⭐ 改成跟圖片一樣的連線資訊
DSN = os.getenv("PG_DSN", "postgresql://postgres:111111@localhost:5432/temple")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "texts"   # <topic>/*.txt|*.md
BATCH_SIZE = int(os.getenv("EMBED_BATCH", "128"))

# 想要的候選模型（依序嘗試）
CANDIDATE_MODELS = [
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-small",
    "sentence-transformers/all-MiniLM-L6-v2",  # 最穩定 fallback（384 維）
]

# 我們希望 DB 一律使用 512 維
TARGET_EMBED_DIM = 512
# ==========================================


# ----------------- 工具 -----------------
def _flatten_supported_models(lst: Iterable) -> List[str]:
    names: List[str] = []
    for item in lst:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict):
            name = item.get("model") or item.get("name") or item.get("model_name")
            if name:
                names.append(name)
    # 去重保序
    seen = set()
    out = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def pick_supported_model() -> str:
    raw = TextEmbedding.list_supported_models()
    supported = _flatten_supported_models(raw)
    print("[info] fastembed 支援的模型（節錄前 20）:", supported[:20])
    for want in CANDIDATE_MODELS:
        if want in supported:
            return want
    if not supported:
        raise RuntimeError("fastembed 沒回傳任何支援模型，請確認安裝。")
    return supported[0]


def stable_uuid(namespace: str, name: str) -> str:
    # 基於固定 namespace + 可讀鍵（topic/filename/idx）產生可重現 UUID
    return str(uuid.uuid5(
        uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8"),
        f"{namespace}::{name}"
    ))


def read_all_text_files(data_dir: Path) -> List[Tuple[str, str, str]]:
    """
    掃描資料夾，回傳 (topic, filename, content)
    """
    files = []
    for p in data_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".txt", ".md"}:
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                print(f"[warn] 讀檔失敗：{p} -> {e}")
                continue
            topic = p.parent.name
            files.append((topic, p.name, content))
    return files


def split_into_chunks(text: str, target_len: int = 400, min_len: int = 120) -> List[str]:
    """
    先依空行分段，再合併成約 200~500 字的片段（避免太碎或過長）。
    """
    raw_blocks = [b.strip() for b in text.replace("\r\n", "\n").split("\n\n") if b.strip()]
    chunks, buf = [], ""
    for block in raw_blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        for ln in lines:
            if len(buf) + len(ln) + 1 <= target_len:
                buf = (buf + " " + ln).strip()
            else:
                if len(buf) >= min_len:
                    chunks.append(buf)
                buf = ln
    if len(buf) >= min_len:
        chunks.append(buf)
    return chunks


def pad_embedding(vec, target_dim: int = TARGET_EMBED_DIM):
    """
    將任意長度的 embedding 補成 target_dim 維（不足補 0，超過就截斷）。
    """
    if vec is None:
        return None
    v = list(vec)
    cur_dim = len(v)
    if cur_dim == target_dim:
        return v
    elif cur_dim > target_dim:
        return v[:target_dim]
    else:
        return v + [0.0] * (target_dim - cur_dim)


def ensure_schema_and_index(conn: psycopg.Connection):
    """
    建立/調整 schema，讓 chunks：
    - 一定有 idx 欄位（圖片可以是 NULL，文字會填 0,1,2,...）
    - embedding 一律為 vector(512)
    不再依照文字模型維度去改 DB 結構。
    """
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # documents 表
        cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
          doc_id UUID PRIMARY KEY,
          topic TEXT,
          filename TEXT,
          media_type TEXT,
          language TEXT,
          text_content TEXT
        );
        """)

        # chunks 表：如果不存在就創一個完整版本
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS chunks (
          chunk_id UUID PRIMARY KEY,
          doc_id UUID REFERENCES documents(doc_id) ON DELETE CASCADE,
          idx INTEGER,
          content TEXT NOT NULL,
          metadata JSONB,
          embedding vector({TARGET_EMBED_DIM}),
          created_at TIMESTAMP DEFAULT now()
        );
        """)

        # 若已經存在（例如之前圖片先建），確保欄位型別正確
        cur.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS idx INTEGER;")
        cur.execute(f"ALTER TABLE chunks ALTER COLUMN embedding TYPE vector({TARGET_EMBED_DIM});")

        # metadata index（圖片那邊可能已經建過，這裡用 IF NOT EXISTS 再保險）
        cur.execute("""
        DO $$
        BEGIN
          IF NOT EXISTS (
            SELECT 1 FROM pg_indexes
            WHERE indexname = 'idx_chunks_metadata'
          ) THEN
            CREATE INDEX idx_chunks_metadata
            ON chunks USING gin (metadata);
          END IF;
        END$$;
        """)

        # 向量索引（如果圖片 ingest_images.py 已建 idx_chunks_embed_ivfflat 也沒關係）
        cur.execute("""
        DO $$
        BEGIN
          IF NOT EXISTS (
            SELECT 1 FROM pg_indexes
            WHERE indexname = 'chunks_embedding_cosine'
          ) THEN
            CREATE INDEX chunks_embedding_cosine
            ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
          END IF;
        END$$;
        """)

        conn.commit()


def upsert_doc_and_chunk(cur: psycopg.Cursor,
                         topic: str,
                         filename: str,
                         idx: int,
                         content: str,
                         vec_padded: List[float]):
    base_key = f"{topic}/{filename}"
    doc_id = stable_uuid("doc", base_key)
    chunk_id = stable_uuid("chunk", f"{base_key}#{idx}")

    meta = {
        "topic": topic,
        "file": filename,
        "source": base_key,
        "index": idx
    }

    # documents
    cur.execute("""
        INSERT INTO documents (doc_id, topic, filename, media_type, language, text_content)
        VALUES (%s, %s, %s, 'text', 'zh', %s)
        ON CONFLICT (doc_id) DO UPDATE
        SET topic = EXCLUDED.topic,
            filename = EXCLUDED.filename,
            text_content = EXCLUDED.text_content;
    """, (doc_id, topic, filename, content[:2000]))

    # chunks
    cur.execute("""
        INSERT INTO chunks (chunk_id, doc_id, idx, content, metadata, embedding)
        VALUES (%s, %s, %s, %s, %s::jsonb, %s::vector)
        ON CONFLICT (chunk_id) DO UPDATE
        SET content   = EXCLUDED.content,
            metadata  = EXCLUDED.metadata,
            embedding = EXCLUDED.embedding;
    """, (chunk_id, doc_id, idx, content, json.dumps(meta, ensure_ascii=False), vec_padded))


# ----------------- 主流程 -----------------
def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"找不到資料夾：{DATA_DIR}")

    print(f"[info] 掃描資料夾：{DATA_DIR}")
    files = read_all_text_files(DATA_DIR)
    if not files:
        print("[warn] 沒發現 .txt/.md 檔案，結束。")
        return

    # 選模型 & 建 embedder
    model_name = pick_supported_model()
    print(f"[info] 使用 fastembed 模型：{model_name}")
    embedder = TextEmbedding(model_name=model_name)

    # 偵測原始向量維度（純顯示用）
    probe = list(embedder.embed(["probe"]))[0]
    raw_dim = len(probe)
    print(f"[info] 模型原始向量維度：{raw_dim}，將補/截成 {TARGET_EMBED_DIM} 維存入 DB")

    print(f"[info] 連線資料庫：{DSN}")
    with psycopg.connect(DSN, row_factory=dict_row) as conn:
        ensure_schema_and_index(conn)

        ok, skip = 0, 0
        with conn.cursor() as cur:
            for topic, filename, content in files:
                chunks = split_into_chunks(content)
                if not chunks:
                    skip += 1
                    print(f"⏭️  略過（無有效段落）：{topic}/{filename}")
                    continue

                # 批次嵌入
                for start in range(0, len(chunks), BATCH_SIZE):
                    batch = chunks[start:start+BATCH_SIZE]
                    vecs = list(embedder.embed(batch))
                    for i, (ch, v) in enumerate(zip(batch, vecs), start=start):
                        vec_padded = pad_embedding(v, TARGET_EMBED_DIM)
                        upsert_doc_and_chunk(cur, topic, filename, i, ch, vec_padded)
                        ok += 1

                print(f"✅ 已寫入：{topic}/{filename}（段落數：{len(chunks)}）")

            conn.commit()

    print(f"\n[done] 寫入 {ok} 筆；略過 {skip} 檔。")


if __name__ == "__main__":
    main()
