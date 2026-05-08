import re
import json
from pathlib import Path
from typing import Dict, List

# 你資料夾常見的「分類尾巴」
SECTION_KEYWORDS = [
    "文化活動",
    "繞境文化",
    "廟宇建築",
    "建築",
    "禁忌",
    "歷史",
    "文化",
]

# 將可能的別名統一成你 Unity 會傳的角色名稱
ENTITY_NORMALIZE = {
    "媽祖娘娘": "媽祖",
    "天上聖母": "媽祖",
    "文昌": "文昌帝君",
    "財神": "財神爺",
}


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_chunks(text: str, max_chars: int = 520, overlap: int = 80) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(n, start + max_chars)
        chunk = text[start:end].strip()

        # 避免太短段落被丟掉（建築名詞有時很短）
        if len(chunk) >= 30:
            chunks.append(chunk)

        if end == n:
            break
        start = max(0, end - overlap)

    return chunks


def parse_folder_to_entity_section(folder_name: str) -> Dict[str, str]:
    """
    例如：
      文昌帝君文化活動 -> entity=文昌帝君, section=文化活動
      媽祖歷史         -> entity=媽祖, section=歷史
      樂成宮建築       -> entity=樂成宮, section=建築
    """
    folder = folder_name.strip()
    section = "其他"
    entity = folder

    # 優先匹配較長的 keyword
    for kw in sorted(SECTION_KEYWORDS, key=len, reverse=True):
        if folder.endswith(kw):
            section = kw
            entity = folder[: -len(kw)].strip() or folder
            break

    # 統一一下 section 命名
    if section == "廟宇建築":
        section = "建築"

    # entity 別名統一
    entity = ENTITY_NORMALIZE.get(entity, entity)

    return {"entity": entity, "section": section}


def build_chunks_jsonl(texts_root: str, out_path: str) -> int:
    """
    ✅ 遞迴掃描 data/texts/**.txt
    ✅ 用「父資料夾名稱」解析出 entity/section
    """
    root = Path(texts_root)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(root.rglob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {texts_root}")

    count = 0
    with out.open("w", encoding="utf-8") as w:
        for txt in txt_files:
            raw = txt.read_text(encoding="utf-8", errors="ignore")
            chunks = split_into_chunks(raw)

            folder_name = txt.parent.name
            parsed = parse_folder_to_entity_section(folder_name)

            entity = parsed["entity"]
            section = parsed["section"]
            topic = folder_name  # topic 用資料夾名最穩

            for i, ch in enumerate(chunks):
                obj = {
                    "content": ch,
                    "metadata": {
                        "source": str(txt).replace("\\", "/"),
                        "file_name": txt.name,
                        "entity": entity,
                        "section": section,
                        "topic": topic,
                        "chunk_id": f"{topic}:{txt.stem}:{i}",
                    },
                }
                w.write(json.dumps(obj, ensure_ascii=False) + "\n")
                count += 1

    return count
