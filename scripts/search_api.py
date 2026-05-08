# -*- coding: utf-8 -*-
# 完整 Temple-RAG + Ollama + 小學生版 API
# 放置於：C:\Users\User\temple-rag\scripts\search_api.py

import os
import json
import psycopg2
import psycopg2.extras
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from typing import List
import requests
import torch
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------
# 基本設定
# ------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:111111@localhost:5432/temple")

# 文本查詢模型（384 維）
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 圖片資料夾（直接給前端用）
IMAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "image")

# WSL 內的 Ollama API
WSL_OLLAMA_URL = "http://172.20.47.58:11434"

# ------------------------------------------------------
# FastAPI 啟動
# ------------------------------------------------------
app = FastAPI(title="Temple-RAG API", version="2.0")

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "..", "templates"))

app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

# ------------------------------------------------------
# 模型初始化
# ------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
text_model = SentenceTransformer(TEXT_MODEL, device=device)


def embed_text(q: str) -> list:
    """產生 384 維向量"""
    v = text_model.encode(q, convert_to_tensor=True, normalize_embeddings=True)
    return v.detach().cpu().tolist()


# ------------------------------------------------------
# Ollama API 呼叫
# ------------------------------------------------------
def call_ollama(prompt: str, model: str = "phi3:mini") -> str:
    """呼叫 WSL 內的 Ollama API"""
    try:
        r = requests.post(
            f"{WSL_OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=60,
        )
        data = r.json()
        return data.get("response", "").strip()
    except Exception as e:
        return f"(小學生版後端錯誤: {e})"


# ------------------------------------------------------
# RAG 查詢 (文本)
# ------------------------------------------------------
def rag_search(query: str, top_k: int = 5):
    """查詢 temple.chunks 裡的文本資料"""
    qv = embed_text(query)

    sql = """
    SELECT 
        chunk_id, content, metadata,
        1 - (embedding <=> %s::vector) AS score
    FROM public.chunks
    WHERE (metadata->>'type') = 'text'
    ORDER BY embedding <=> %s::vector
    LIMIT %s;
    """

    with psycopg2.connect(DATABASE_URL) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (qv, qv, top_k))
            rows = cur.fetchall()

    return rows


# ------------------------------------------------------
# 首頁（使用 home.html）
# ------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


# ------------------------------------------------------
# 健康檢查
# ------------------------------------------------------
@app.get("/healthz")
def healthz():
    return {"ok": True}


# ------------------------------------------------------
# 文本 RAG 查詢（專業版）
# ------------------------------------------------------
@app.post("/ask")
def ask(payload: dict):
    question = payload.get("question", "")
    top_k = payload.get("top_k", 5)

    rows = rag_search(question, top_k)

    # 專業版答案 = 把 top-k 的內容串起來
    answer = "\n".join([r["content"] for r in rows]) if rows else "查無相關資料"

    sources = []
    for r in rows:
        md = r["metadata"]
        sources.append({
            "file": md.get("file"),
            "topic": md.get("topic"),
            "index": md.get("index"),
            "score": float(r["score"]),
        })

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }


# ------------------------------------------------------
# 文本 RAG → 小學生版（Ollama）
# ------------------------------------------------------
@app.post("/ask-kid")
def ask_kid(payload: dict):
    question = payload.get("question", "")
    top_k = payload.get("top_k", 5)

    # RAG 先抓資料
    rows = rag_search(question, top_k)
    rag_text = "\n".join([r["content"] for r in rows]) if rows else "查無資料"

    # 給 Ollama 的 prompt（優化過）
    prompt = f"""
你是一位台灣國小老師，請用非常簡單、易懂、短句的方式回答問題。

【學生的問題】
{question}

【給老師參考的資料】
{rag_text}

請將內容整理成 3～5 句話，
不能太長，
不能使用生硬的宗教術語，
要像在跟小學生聊天一樣，
請開始回答：
"""

    kid_text = call_ollama(prompt)

    return {
        "question": question,
        "kid_answer": kid_text,
        "sources": [r["metadata"] for r in rows],
    }


# ------------------------------------------------------
# 圖片搜尋（使用 CLIP-text-to-image）
# ------------------------------------------------------
@app.get("/search/images")
def search_images(
    query: str = Query(...),
    top_k: int = 6,
):
    """文字 → 找相似寺廟建築圖片"""
    qv = embed_text(query)

    sql = """
    SELECT 
        chunk_id, content, metadata,
        1 - (embedding <=> %s::vector) AS score
    FROM public.chunks
    WHERE (metadata->>'type') = 'image'
    ORDER BY embedding <=> %s::vector
    LIMIT %s;
    """

    with psycopg2.connect(DATABASE_URL) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (qv, qv, top_k))
            rows = cur.fetchall()

    results = []
    for r in rows:
        md = r["metadata"]
        results.append({
            "filename": md.get("filename"),
            "category": md.get("category"),
            "score": float(r["score"]),
            "path": f"/images/{md.get('category')}/{md.get('filename')}{md.get('ext')}"
        })

    return {"query": query, "results": results}


# ------------------------------------------------------
# 下載圖片（自動處理路徑）
# ------------------------------------------------------
@app.get("/images/{category}/{filename}")
def get_image(category: str, filename: str):
    path = os.path.join(IMAGE_DIR, category, filename)
    if os.path.exists(path):
        return FileResponse(path)
    return JSONResponse({"error": "not found"}, status_code=404)
