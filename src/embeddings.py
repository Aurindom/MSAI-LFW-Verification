import numpy as np
import torch
from PIL import Image

_model = None


def _get_model():
    global _model
    if _model is None:
        from facenet_pytorch import InceptionResnetV1
        _model = InceptionResnetV1(pretrained="vggface2").eval()
    return _model


def preprocess(path: str, image_size: int = 160) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((image_size, image_size))
    arr = np.array(img, dtype=np.float32)
    tensor = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor / 127.5 - 1.0


def embed(tensor: torch.Tensor) -> np.ndarray:
    model = _get_model()
    with torch.no_grad():
        emb = model(tensor)
    return emb.squeeze(0).numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
