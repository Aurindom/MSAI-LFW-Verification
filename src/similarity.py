import numpy as np


# =========================
# VECTORIZE IMPLEMENTATIONS
# =========================

def cosine_similarity_vectorized(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = np.sum(a * b, axis=1)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    return dot / (norm_a * norm_b)


def euclidean_distance_vectorized(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=1)


# =========================
# LOOP BASELINES (FOR BENCH)
# =========================

def cosine_similarity_loop(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    results = []
    for i in range(a.shape[0]):
        dot = np.dot(a[i], b[i])
        norm = np.linalg.norm(a[i]) * np.linalg.norm(b[i])
        results.append(dot / norm)
    return np.array(results)


def euclidean_distance_loop(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    results = []
    for i in range(a.shape[0]):
        results.append(np.linalg.norm(a[i] - b[i]))
    return np.array(results)