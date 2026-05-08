import os
import sys
import json
import uuid
import pathlib
import psycopg2
import psycopg2.extras
from typing import List
from PIL import Image

import torch
from transformers import CLIPProcessor, CLIPModel



# =========================
# 基本設定
# =========================

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]

IMAGE_ROOT = str(BASE_DIR / "data" / "image")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:111111@localhost:5432/temple"
)

# ⭐ 官方原版 CLIP（512 維）
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"



# =========================
# DB Tools
# =========================

def ensure_pgvector(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()


def ensure_chunks_index(conn):
    with conn.cursor() as cur:
        cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_indexes
                WHERE indexname='idx_chunks_embed_ivfflat'
            ) THEN
                CREATE INDEX idx_chunks_embed_ivfflat
                ON public.chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists=100);
            END IF;
        END $$;
        """)
        conn.commit()



# =========================
# 圖片列表
# =========================

def list_images(root: str) -> List[pathlib.Path]:
    EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    files = []
    for p in pathlib.Path(root).rglob("*"):
        if p.is_file() and p.suffix.lower() in EXT:
            files.append(p)
    return files



# =========================
# CLIP 圖像 → embedding
# =========================

def image_to_embedding(processor, model, pil_img: Image.Image):
    """
    PIL Image → CLIP 512 維 embedding
    """
    inputs = processor(images=pil_img, return_tensors="pt", padding=True)

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.get_image_features(**inputs)

    # L2 normalize
    emb = outputs / outputs.norm(p=2, dim=-1, keepdim=True)

    return emb.squeeze(0)



# =========================
# Upsert DB
# =========================

def upsert_chunk(conn, *, path_abs, category, filename, ext, embedding):

    metadata = {
        "type": "image",
        "path_abs": path_abs,
        "category": category,
        "filename": filename,
        "ext": ext
    }

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:

        # 避免重複
        cur.execute("""
            SELECT chunk_id FROM public.chunks
            WHERE metadata->>'path_abs' = %s
            LIMIT 1;
        """, (path_abs,))
        found = cur.fetchone()

        if found:
            cur.execute("""
                UPDATE public.chunks
                SET embedding = %s::vector,
                    metadata = %s::jsonb
                WHERE chunk_id = %s;
            """, (embedding, json.dumps(metadata), found["chunk_id"]))

        else:
            chunk_id = str(uuid.uuid4())
            content = filename

            cur.execute("""
                INSERT INTO public.chunks (chunk_id, doc_id, content, metadata, embedding, created_at)
                VALUES (%s, NULL, %s, %s::jsonb, %s::vector, NOW());
            """, (chunk_id, content, json.dumps(metadata), embedding))

    conn.commit()



# =========================
# 主程式
# =========================

def main():
    print(f"[INGEST] IMAGE_ROOT={IMAGE_ROOT}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INGEST] device={device}")

    # ⭐ 正版 CLIP，絕對可用
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, use_safetensors=True).to(device)

    conn = psycopg2.connect(DATABASE_URL)
    ensure_pgvector(conn)
    ensure_chunks_index(conn)

    imgs = list_images(IMAGE_ROOT)
    print(f"[INGEST] Found {len(imgs)} images")

    for p in imgs:
        rel = p.relative_to(IMAGE_ROOT)
        category = rel.parts[0] if len(rel.parts) >= 2 else ""

        filename = p.stem
        ext = p.suffix
        path_abs = str(p.resolve())

        try:
            pil = Image.open(p).convert("RGB")
            emb = image_to_embedding(processor, model, pil)

            upsert_chunk(
                conn,
                path_abs=path_abs,
                category=category,
                filename=filename,
                ext=ext,
                embedding=emb.detach().cpu().tolist()
            )

            print(f"[OK] {path_abs}")

        except Exception as e:
            print(f"[SKIP] {path_abs} -> {e}")
            conn.rollback()

    conn.close()
    print("[DONE] image ingestion completed.")



if __name__ == "__main__":
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except:
            pass
    main()
