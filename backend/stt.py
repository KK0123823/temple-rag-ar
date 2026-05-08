from __future__ import annotations

import os
import re
import tempfile
import wave
import asyncio
from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel

from faster_whisper import WhisperModel

router = APIRouter()


class STTTextRequest(BaseModel):
    transcript: str


def normalize_choice(text: str) -> str:
    if not text:
        return ""

    raw = text.strip()
    t = raw.upper()
    t = re.sub(r"[\s\.\,\!\?\-_/\\]+", "", t)

    if "A" in t:
        return "A"
    if "B" in t:
        return "B"
    if "C" in t:
        return "C"

    if "第一" in raw or "1" in t:
        return "A"
    if "第二" in raw or "2" in t:
        return "B"
    if "第三" in raw or "3" in t:
        return "C"

    if "ㄟ" in raw or "欸" in raw or "誒" in raw:
        return "A"

    return ""


@router.post("/stt_text")
def stt_text(req: STTTextRequest):
    choice = normalize_choice(req.transcript)
    return {"ok": True, "choice": choice, "raw": req.transcript}


# -----------------------------
# ✅ 全域單例模型 + 單工鎖
# -----------------------------
MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")

# Windows + GPU：用 float16
# 如果你遇到奇怪當機，可改成 int8_float16（更穩但稍慢）
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")

_model: WhisperModel | None = None
_model_lock = asyncio.Lock()
_infer_lock = asyncio.Lock()

CHOICE_PROMPT = "請只回答 A、B 或 C（或第一/第二/第三）。"


def _ensure_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(
            MODEL_SIZE,
            device="cuda",
            compute_type=COMPUTE_TYPE
        )
    return _model


def _wav_duration_seconds(path: str) -> float:
    try:
        with wave.open(path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate <= 0:
                return 0.0
            return frames / float(rate)
    except Exception:
        return 0.0


@router.on_event("startup")
async def _warmup():
    # ✅ 啟動時先載入+暖機一次，避免第一次卡太久
    async with _model_lock:
        model = _ensure_model()

    # 做一次很短的空暖機（不一定需要真音檔）
    # 這邊不跑 transcribe（避免沒音檔），只確保模型完成 init
    # 如果你想更徹底，可用一段短 wav 做 transcribe
    return


@router.post("/stt")
async def stt(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    if not audio_bytes:
        return {"text": "", "choice": "", "error": "Empty audio"}

    tmp_path = None
    try:
        # 1) 寫入暫存檔
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        dur = _wav_duration_seconds(tmp_path)
        if dur < 0.35:
            return {"text": "", "choice": "", "error": f"Audio too short ({dur:.2f}s)."}

        # 2) 模型只初始化一次（鎖住避免同時 init）
        async with _model_lock:
            model = _ensure_model()

        # 3) ✅ 同一時間只允許一個推論（最重要：防止第2次卡死）
        async with _infer_lock:
            segments, info = model.transcribe(
                tmp_path,
                language="zh",
                vad_filter=True,
                beam_size=3,
                best_of=3,
                temperature=0.0,
                condition_on_previous_text=False,
                initial_prompt=CHOICE_PROMPT
            )

        transcript = "".join(seg.text for seg in segments).strip()

        # 逗號保護（通常代表靜音/太小聲）
        if transcript in {",", "，", ".", "。", "…", "..."}:
            return {"text": transcript, "choice": "", "error": "Only punctuation; audio likely silent/too low."}

        choice = normalize_choice(transcript)
        return {"text": transcript, "choice": choice, "error": ""}

    except Exception as e:
        return {"text": "", "choice": "", "error": str(e)}

    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
