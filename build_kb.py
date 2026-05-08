# build_kb.py
from pathlib import Path
from backend.utils_text import build_chunks_jsonl

def main():
    project_root = Path(__file__).resolve().parent
    kb_dir = project_root / "data" / "kb"
    texts_root = project_root / "data" / "texts"
    out_path = kb_dir / "chunks.jsonl"

    kb_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] 開始處理原始文本...")
    n = build_chunks_jsonl(
        texts_root=str(texts_root),
        out_path=str(out_path),
    )
    print(f"[OK] chunks.jsonl 產生完成，共 {n} 段 (存放於 {out_path})")

if __name__ == "__main__":
    main()
