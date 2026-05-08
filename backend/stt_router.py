# stt_router.py
from __future__ import annotations

import asyncio
from fastapi import APIRouter, UploadFile, File
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
import time

from .stt_worker import run_worker  # 引用上面的 worker

router = APIRouter()

_worker_proc: Process | None = None
_parent_conn: Connection | None = None
_worker_lock = asyncio.Lock()


def _start_worker():
    global _worker_proc, _parent_conn
    if _worker_proc is not None and _worker_proc.is_alive():
        return

    parent_conn, child_conn = Pipe(duplex=True)
    p = Process(target=run_worker, args=(child_conn,), daemon=True)
    p.start()
    _worker_proc = p
    _parent_conn = parent_conn


def _stop_worker():
    global _worker_proc, _parent_conn
    try:
        if _parent_conn is not None:
            try:
                _parent_conn.send(None)
            except Exception:
                pass
    except Exception:
        pass

    try:
        if _worker_proc is not None and _worker_proc.is_alive():
            _worker_proc.terminate()
    except Exception:
        pass

    _worker_proc = None
    _parent_conn = None


@router.on_event("startup")
async def on_startup():
    _start_worker()


@router.on_event("shutdown")
async def on_shutdown():
    _stop_worker()


# 調整重點：確保 Timeout 設定與 Unity 端匹配
@router.post("/stt")
async def stt(file: UploadFile = File(...)):
    # 增加檔案大小檢查，避免接收到空檔
    audio_bytes = await file.read()
    if len(audio_bytes) < 100:
        return {"text": "", "choice": "", "error": "Received file too small"}

    async with _worker_lock:
        _start_worker()
        try:
            _parent_conn.send({"audio_bytes": audio_bytes})
            
            # 給予充足的辨識時間（GPU small 模型通常在 1-2 秒內，但第一次加載會久一點）
            timeout_sec = 20.0 
            t0 = time.time()
            while True:
                if _parent_conn.poll(0.1):
                    return _parent_conn.recv()
                if time.time() - t0 > timeout_sec:
                    _stop_worker() # 逾時強制重啟，防止 CUDA 殭屍進程
                    return {"text": "", "choice": "", "error": "GPU Inference Timeout"}
                await asyncio.sleep(0.05) # 釋放 CPU 給其他協程
        except Exception as e:
            _stop_worker()
            return {"text": "", "choice": "", "error": str(e)}