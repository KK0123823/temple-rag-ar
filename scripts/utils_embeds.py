import numpy as np, torch, open_clip
from sentence_transformers import SentenceTransformer
from PIL import Image

# 文字向量（768維）
_text = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
def embed_text(text: str) -> np.ndarray:
    v = _text.encode([text], normalize_embeddings=True)[0]
    return v.astype("float32")

# 影像向量（OpenCLIP ViT-B/32，768維）
_device = "cuda" if torch.cuda.is_available() else "cpu"
_model, _, _preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
_model = _model.to(_device).eval()

@torch.no_grad()
def embed_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    x = _preprocess(img).unsqueeze(0).to(_device)
    feat = _model.encode_image(x)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).detach().cpu().numpy().astype("float32")
