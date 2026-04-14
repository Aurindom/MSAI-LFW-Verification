import numpy as np
import pandas as pd
from PIL import Image


def _load_vector(path: str, image_size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((image_size, image_size))
    vec = np.array(img, dtype=np.float32).flatten()
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def compute_pixel_similarity(path_a: str, path_b: str, image_size: int = 64) -> float:
    va = _load_vector(path_a, image_size)
    vb = _load_vector(path_b, image_size)
    cosine = float(np.dot(va, vb))
    return (cosine + 1.0) / 2.0


def score_pairs(pairs_df: pd.DataFrame, image_size: int = 64) -> np.ndarray:
    scores = np.array([
        compute_pixel_similarity(row.left_path, row.right_path, image_size)
        for row in pairs_df.itertuples(index=False)
    ])
    return scores
