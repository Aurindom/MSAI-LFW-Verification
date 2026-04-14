import os

import numpy as np
import pytest
from PIL import Image

from src.embeddings import cosine_similarity, embed, preprocess
from src.inference import InferenceResult, compute_confidence, verify_pair


def _make_image(path: str, color: tuple) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.full((160, 160, 3), color, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


@pytest.fixture(scope="module")
def sample_images(tmp_path_factory):
    base = tmp_path_factory.mktemp("imgs")
    p_a0 = str(base / "id_a" / "a_0.jpg")
    p_a1 = str(base / "id_a" / "a_1.jpg")
    p_b0 = str(base / "id_b" / "b_0.jpg")
    _make_image(p_a0, (200, 100, 50))
    _make_image(p_a1, (198, 102, 51))
    _make_image(p_b0, (30, 200, 180))
    return {"a0": p_a0, "a1": p_a1, "b0": p_b0}


class TestPreprocess:

    def test_output_shape(self, sample_images):
        t = preprocess(sample_images["a0"], image_size=160)
        assert t.shape == (1, 3, 160, 160)

    def test_pixel_range(self, sample_images):
        t = preprocess(sample_images["a0"], image_size=160)
        assert float(t.min()) >= -1.0
        assert float(t.max()) <= 1.0

    def test_deterministic(self, sample_images):
        t1 = preprocess(sample_images["a0"])
        t2 = preprocess(sample_images["a0"])
        import torch
        assert torch.equal(t1, t2)


class TestEmbed:

    def test_output_shape(self, sample_images):
        t = preprocess(sample_images["a0"])
        emb = embed(t)
        assert emb.shape == (512,)

    def test_output_dtype(self, sample_images):
        t = preprocess(sample_images["a0"])
        emb = embed(t)
        assert emb.dtype == np.float32

    def test_deterministic(self, sample_images):
        t = preprocess(sample_images["a0"])
        e1 = embed(t)
        e2 = embed(t)
        np.testing.assert_array_equal(e1, e2)


class TestCosineSimilarity:

    def test_identical_vectors(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        v = np.array([1.0, 0.0], dtype=np.float32)
        assert cosine_similarity(v, -v) == pytest.approx(-1.0)

    def test_zero_vector(self):
        v = np.zeros(3, dtype=np.float32)
        u = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert cosine_similarity(v, u) == 0.0


class TestComputeConfidence:

    def test_at_boundary_same(self):
        assert compute_confidence(0.5, threshold=0.5) == pytest.approx(0.0)

    def test_at_boundary_diff(self):
        assert compute_confidence(0.5, threshold=0.5) == pytest.approx(0.0)

    def test_max_same(self):
        assert compute_confidence(1.0, threshold=0.5) == pytest.approx(1.0)

    def test_max_diff(self):
        assert compute_confidence(0.0, threshold=0.5) == pytest.approx(1.0, abs=1e-6)

    def test_midpoint_same(self):
        conf = compute_confidence(0.75, threshold=0.5)
        assert conf == pytest.approx(0.5)

    def test_range(self):
        for thr in [0.3, 0.5, 0.7]:
            for score in np.linspace(0.01, 0.99, 20):
                c = compute_confidence(float(score), thr)
                assert 0.0 <= c <= 1.0


class TestVerifyPair:

    def test_result_type(self, sample_images):
        result = verify_pair(sample_images["a0"], sample_images["a1"], threshold=0.5)
        assert isinstance(result, InferenceResult)

    def test_score_in_range(self, sample_images):
        result = verify_pair(sample_images["a0"], sample_images["b0"], threshold=0.5)
        assert 0.0 <= result.score <= 1.0

    def test_confidence_in_range(self, sample_images):
        result = verify_pair(sample_images["a0"], sample_images["b0"], threshold=0.5)
        assert 0.0 <= result.confidence <= 1.0

    def test_decision_matches_threshold(self, sample_images):
        result = verify_pair(sample_images["a0"], sample_images["b0"], threshold=0.5)
        if result.score >= 0.5:
            assert result.decision == "SAME"
        else:
            assert result.decision == "DIFFERENT"

    def test_latency_positive(self, sample_images):
        result = verify_pair(sample_images["a0"], sample_images["b0"], threshold=0.5)
        assert result.latency_s > 0

    def test_stage_latencies_positive(self, sample_images):
        result = verify_pair(sample_images["a0"], sample_images["b0"], threshold=0.5)
        assert result.preprocess_s >= 0
        assert result.embed_s >= 0
        assert result.score_s >= 0


class TestSmokeTest:

    def test_cli_inference_path_completes(self, sample_images):
        result = verify_pair(
            sample_images["a0"],
            sample_images["a1"],
            threshold=0.5,
            image_size=160,
        )
        assert result.decision in ("SAME", "DIFFERENT")
        assert isinstance(result.score, float)
        assert isinstance(result.confidence, float)
        assert isinstance(result.latency_s, float)
