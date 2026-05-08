import os
import time
import wave
import re
import sys
import numpy as np
from multiprocessing.connection import Connection
from faster_whisper import WhisperModel

# --- 1. 深度 DLL 路徑修復 (針對 Windows + CUDA 12) ---
if sys.platform == "win32":
    # 搜尋虛擬環境中所有可能的 NVIDIA DLL 目錄
    venv_base = os.path.join(os.getcwd(), ".venv", "Lib", "site-packages", "nvidia")
    possible_bins = [
        os.path.join(venv_base, "cublas", "bin"),
        os.path.join(venv_base, "cudnn", "bin"),
        os.path.join(venv_base, "cuda_runtime", "bin")
    ]
    for p in possible_bins:
        if os.path.exists(p):
            # 同時加入環境變數與 DLL 目錄，確保 ctranslate2 讀得到
            os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]
            try:
                os.add_dll_directory(p)
                print(f"已掛載 DLL 路徑: {p}")
            except Exception:
                pass

# --- 2. 除錯資料夾配置 ---
CURRENT_DIR = os.getcwd() 
DEBUG_SAVE_DIR = os.path.join(CURRENT_DIR, "debug_audio_logs")
if not os.path.exists(DEBUG_SAVE_DIR):
    os.makedirs(DEBUG_SAVE_DIR)

def normalize_choice(text: str) -> str:
    if not text: return ""
    raw = text.strip()
    t = re.sub(r"[\s\.\,\!\?\-_/\\]+", "", raw.upper())
    # 增加更多判斷語句，涵蓋常見錄音偏差
    if any(x in t for x in ["A", "第一", "1", "欸", "ㄟ", "誒", "選A"]): return "A"
    if any(x in t for x in ["B", "第二", "2", "必", "逼", "選B"]): return "B"
    if any(x in t for x in ["C", "第三", "3", "吸", "西", "選C"]): return "C"
    return ""

def run_worker(conn: Connection):
    print("--- [Worker] 模型初始化中... ---")
    model_size = os.getenv("WHISPER_MODEL_SIZE", "small")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
    
    # 初始化模型
    def load_model(use_gpu=True):
        if use_gpu:
            try:
                print("嘗試啟動 GPU (CUDA)...")
                return WhisperModel(model_size, device="cuda", compute_type=compute_type)
            except Exception as e:
                print(f"GPU 啟動失敗: {e}，改用 CPU。")
        return WhisperModel(model_size, device="cpu", compute_type="int8")

    model = load_model(use_gpu=True)
    print(f"--- [Worker] 辨識引擎已就緒 ---")

    while True:
        try:
            msg = conn.recv()
            if msg is None: break
            audio_bytes = msg.get("audio_bytes", b"")
            
            # 儲存音訊供後續手動聽取
            timestamp = int(time.time() * 1000)
            file_path = os.path.join(DEBUG_SAVE_DIR, f"rec_{timestamp}.wav")
            with open(file_path, "wb") as f:
                f.write(audio_bytes)

            with wave.open(file_path, "rb") as wf:
                data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                rms = np.sqrt(np.mean(data.astype(np.float32)**2)) if len(data) > 0 else 0
                duration = wf.getnframes() / wf.getframerate()

            print(f"收到音訊 | RMS: {rms:.2f} | 時長: {duration:.2f}s")

            if rms < 10:
                conn.send({"text": "", "choice": "", "error": "音量太小"})
                continue

            # 嘗試推論，如果 DLL 報錯就當場切換成 CPU 並重試
            try:
                segments, _ = model.transcribe(
                    file_path, language="zh", vad_filter=False, 
                    initial_prompt="A, B, C, 第一, 第二, 第三, 媽祖, 文昌帝君, 財神爺, 建築, 廟宇, 選項"
                )
                transcript = "".join(s.text for s in segments).strip()
            except RuntimeError as e:
                if "library cublas64" in str(e).lower() or "not found" in str(e).lower():
                    print("GPU 推論失敗 (DLL 缺失)，緊急切換至 CPU...")
                    model = load_model(use_gpu=False)
                    # 使用 CPU 重試一次
                    segments, _ = model.transcribe(file_path, language="zh", vad_filter=False)
                    transcript = "".join(s.text for s in segments).strip()
                else:
                    raise e

            # 處理結果
            if not transcript or transcript in {",", "，", ".", "。"}:
                conn.send({"text": "", "choice": "", "error": "辨識為空"})
            else:
                choice = normalize_choice(transcript)
                print(f"辨識成功: '{transcript}' -> {choice}")
                conn.send({"text": transcript, "choice": choice, "error": ""})

        except Exception as e:
            print(f"Worker 嚴重錯誤: {e}")
            conn.send({"text": "", "choice": "", "error": str(e)})