import time
from dataclasses import dataclass

import numpy as np

from src.embeddings import cosine_similarity, embed, preprocess


@dataclass
class InferenceResult:
    img_a: str
    img_b: str
    score: float
    threshold: float
    decision: str
    confidence: float
    latency_s: float
    preprocess_s: float
    embed_s: float
    score_s: float


def compute_confidence(score: float, threshold: float) -> float:
    if threshold <= 0.0:
        return 1.0
    if threshold >= 1.0:
        return 1.0
    if score >= threshold:
        room = 1.0 - threshold
        return float((score - threshold) / room) if room > 0 else 1.0
    else:
        return float((threshold - score) / threshold)


def verify_pair(
    img_path_a: str,
    img_path_b: str,
    threshold: float,
    image_size: int = 160,
) -> InferenceResult:
    t_start = time.perf_counter()

    t0 = time.perf_counter()
    tensor_a = preprocess(img_path_a, image_size)
    tensor_b = preprocess(img_path_b, image_size)
    preprocess_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    emb_a = embed(tensor_a)
    emb_b = embed(tensor_b)
    embed_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    score = cosine_similarity(emb_a, emb_b)
    score = float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))
    score_s = time.perf_counter() - t0

    decision = "SAME" if score >= threshold else "DIFFERENT"
    confidence = compute_confidence(score, threshold)
    latency_s = time.perf_counter() - t_start

    return InferenceResult(
        img_a=img_path_a,
        img_b=img_path_b,
        score=round(score, 6),
        threshold=threshold,
        decision=decision,
        confidence=round(confidence, 6),
        latency_s=round(latency_s, 6),
        preprocess_s=round(preprocess_s, 6),
        embed_s=round(embed_s, 6),
        score_s=round(score_s, 6),
    )
