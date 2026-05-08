from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Temple RAG API Test")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "ok": True,
        "message": "Render API is running"
    }

@app.get("/health")
def health():
    return {
        "status": "ok"
    }

@app.post("/ask")
def ask(data: dict):
    question = data.get("query") or data.get("question") or ""

    return {
        "ok": True,
        "kids": "Render 外網部署成功，AR 已經可以呼叫 API。",
        "panel": f"收到問題：{question}",
        "rag_used": False
    }