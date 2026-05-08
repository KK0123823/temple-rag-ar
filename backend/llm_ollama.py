# backend/llm_ollama.py
from __future__ import annotations

from typing import Dict, List, Tuple
import json
import re
import subprocess
from opencc import OpenCC

OLLAMA_MODEL = "qwen2.5"
cc = OpenCC("s2twp")

PERSONA: Dict[str, str] = {
    "mazu": "你是台灣廟宇中的媽祖本人。語氣慈悲溫暖，但內容要有文化深度與教育性。",
    "wenchang": "你是文昌帝君本人。語氣儒雅親切，善於用例子解釋並引導學習。",
    "caishen": "你是財神爺本人。語氣正向有精神，但要講清楚信仰象徵與正確觀念。",
    "default": "你是樂成宮的資深導覽廟公。語氣親切專業，擅長把建築與民俗講得清楚好懂。",
}


def _truncate_zh(text: str, max_chars: int) -> str:
    """安全截斷：盡量在句號/逗號處切，避免 TTS 太長。"""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text

    cut = text[:max_chars]
    # 往前找最近的結尾標點
    for p in ["。", "！", "？", "；", "，"]:
        idx = cut.rfind(p)
        if idx >= max_chars * 0.6:  # 不要切太前面
            return cut[: idx + 1].strip()

    return cut.strip()


def _compact_one_line(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\n", " ")).strip()


def generate_kids_and_detail(query: str, ctx_text: str, persona_key: str) -> Tuple[str, str]:
    """
    回傳：
    - kids：給語音播放用（短，2~3句）
    - panel：給面板閱讀用（較完整，但仍限制）
    """
    persona = PERSONA.get(persona_key, PERSONA["default"])
    has_ctx = len((ctx_text or "").strip()) > 0

    mode = "【無資料】資料庫沒有直接命中。請用你作為該角色的通用專業知識回答。"
    if has_ctx:
        mode = "【資料優先】請以資料內容為主回答，不可新增資料沒有的具體事實。"

    prompt = f"""
{persona}
{mode}

【任務】
你要以第一人稱(我)教導小朋友理解廟宇建築與文化：好懂但不幼稚，要能學到專業知識。

【輸出兩段文字】
1) kids：給語音播放用，最多 2~3 句，總長 <= 110 個中文字，不要換行。
2) panel：給面板閱讀用，最多 5 句，總長 <= 260 個中文字，可以換行但不要太長。

【內容要求】
- kids：1句白話定義 + 1句用途/意義（最多再加1句補充）
- panel：
  - 白話解釋 1~2 句
  - 至少 1 個專業名詞（括號解釋）
  - 1 句用途/文化意義/常見誤解澄清

【資料遵循規則（非常重要）】
- 若「可參考資料」不為空：回答必須以資料為主，不可新增資料沒有提到的具體事實。
- panel 內必須引用資料中的至少 1 句原句（可微調標點，不可改意）。
- 若資料不足以回答某部分，請明確說「資料沒有提到」，再用「補充」方式以常識補足。

【禁止事項】
不要提到「資料庫、檢索、向量、模型、系統」等字眼。

【輸出格式】
只輸出 JSON：
{{"kids":"...","panel":"..."}}

【問題】
{query}

【可參考資料】
{ctx_text if has_ctx else "（無資料）"}
""".strip()

    raw = _ollama_run(prompt)

    # 容錯：抽出 JSON
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        kids = cc.convert(raw.strip())
        kids = _compact_one_line(kids)
        kids = _truncate_zh(kids, 110)
        return kids, ""

    js = m.group(0)
    try:
        obj = json.loads(js)
        kids = cc.convert(str(obj.get("kids", ""))).strip()
        panel = cc.convert(str(obj.get("panel", ""))).strip()

        kids = _compact_one_line(kids)
        kids = _truncate_zh(kids, 110)

        # panel 可換行，但仍要限制長度
        panel = panel.strip()
        panel = _truncate_zh(panel, 260)

        if not kids:
            kids = cc.convert(raw.strip())
            kids = _compact_one_line(kids)
            kids = _truncate_zh(kids, 110)

        return kids, panel
    except Exception:
        kids = cc.convert(raw.strip())
        kids = _compact_one_line(kids)
        kids = _truncate_zh(kids, 110)
        return kids, ""


def _ollama_run(prompt: str) -> str:
    p = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    return p.stdout.decode("utf-8", errors="ignore").strip()


def build_ctx_numbered(contexts: List[dict]) -> Tuple[str, List[str]]:
    lines = []
    cites = []
    for i, c in enumerate(contexts):
        lines.append(f"[{i+1}] {str(c.get('content','')).strip()}")
        fname = (c.get("metadata", {}) or {}).get("file_name", "unknown")
        cites.append(f"[{i+1}] {fname}")
    return "\n\n".join(lines), cites


def sanitize_citations_in_text(text: str, max_n: int) -> str:
    if max_n <= 0:
        return re.sub(r"\[\d+\]", "", text)

    def keep(m):
        n = int(m.group(1))
        return f"[{n}]" if 1 <= n <= max_n else ""

    return re.sub(r"\[(\d+)\]", keep, text)
