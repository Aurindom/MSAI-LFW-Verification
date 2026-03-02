import math
import numpy as np


def cosine_loop(a: np.ndarray, b: np.ndarray) -> list:
    a_l, b_l = a.tolist(), b.tolist()
    out = []
    for ai, bi in zip(a_l, b_l):
        dot = sum(x * y for x, y in zip(ai, bi))
        norm = (sum(x * x for x in ai) ** 0.5) * (sum(x * x for x in bi) ** 0.5)
        out.append(dot / norm if norm > 0 else 0.0)
    return out


def euclidean_loop(a: np.ndarray, b: np.ndarray) -> list:
    a_l, b_l = a.tolist(), b.tolist()
    out = []
    for ai, bi in zip(a_l, b_l):
        out.append(math.sqrt(sum((x - y) ** 2 for x, y in zip(ai, bi))))
    return out


def cosine_vectorized(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dots = np.sum(a * b, axis=1)
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.where(norms > 0, dots / norms, 0.0)


def euclidean_vectorized(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=1)
