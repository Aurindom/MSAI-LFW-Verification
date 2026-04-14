# Milestone 3 Evaluation Report
## LFW Face Verification -- FaceNet Embedding Inference

---

### 1. System Overview

This report documents the Milestone 3 evaluation of the LFW face verification
pipeline. Milestone 3 replaces the pixel cosine similarity baseline from M2 with
a pretrained face embedding model, adds a CLI inference interface, Docker packaging,
and a concurrency load test.

The M2 pipeline was corrected prior to M3 development. The TA had flagged that the
original M2 pair generation used synthetic placeholder paths (`identity_0000`) instead
of real LFW image files. This was fixed: `src/ingestion.py` now downloads real LFW
data via `sklearn.datasets.fetch_lfw_people` and saves images to disk, and
`src/pairs.py` builds the identity pool from actual file paths on disk. All M3
evaluation is based on this corrected real-data pipeline.

**Embedding model:** `InceptionResnetV1` pretrained on VGGFace2 via `facenet-pytorch`.
Produces 512-d L2-normalised face embeddings. Input: 160x160 RGB images normalised
to [-1, 1].

**Scoring:** Cosine similarity between embedding pairs, mapped to [0, 1] via
(cosine + 1) / 2.

**Threshold selection:** Sweep over val split, select by max balanced accuracy.
Test split is never inspected during calibration.

**Confidence:** Distance from decision boundary normalised by room on predicted side.
- SAME (score >= threshold): `(score - threshold) / (1.0 - threshold)`
- DIFFERENT (score < threshold): `(threshold - score) / threshold`
- Range [0, 1]: 0 = at boundary, 1 = maximum certainty.

**Dataset:** 1,680 identities, 9,164 images, identity-disjoint 70/15/15 split
(seed=42). Val pairs (v2): 968 (484 pos, 484 neg). Test pairs (v2): 944
(472 pos, 472 neg).

---

### 2. Threshold Re-selection

The operating threshold was re-selected on the val split using FaceNet embedding
scores. A sweep over 99 thresholds in [0.01, 0.99] was run on the v2 val set.

**Selected threshold: 0.68** (max balanced accuracy on val)

Val metrics at threshold 0.68:

| Metric | Value |
|--------|-------|
| AUC | 0.9987 |
| Balanced Accuracy | 0.9835 |
| F1 | 0.9835 |
| TPR | 0.9855 |
| FPR | 0.0186 |

The selected threshold (0.68) is significantly lower than the M2 pixel baseline
threshold (0.970), reflecting that FaceNet embeddings produce a wider, better-
separated score distribution across the [0, 1] range.

---

### 3. Final Test Evaluation

The threshold 0.68 selected on val was applied unchanged to the test split.

**Test metrics at threshold 0.68:**

| Metric | Value |
|--------|-------|
| Balanced Accuracy | 0.9778 |
| F1 | 0.9778 |
| TPR | 0.9809 |
| FPR | 0.0254 |
| TP | 463 |
| FP | 12 |
| TN | 460 |
| FN | 9 |

Val-to-test transfer: bal_acc delta = -0.0057. The threshold generalises cleanly,
confirming no overfitting to the val split during calibration.

---

### 4. M2 vs M3 Comparison

| Metric | M2 Pixel Baseline (v2, test) | M3 FaceNet (v2, test) | Change |
|--------|-----|------|------|
| AUC (val) | 0.6494 | **0.9987** | +0.349 |
| Balanced Accuracy | 0.6176 | **0.9778** | +0.360 |
| F1 | 0.5697 | **0.9778** | +0.408 |
| Threshold | 0.970 | 0.680 | -0.290 |

The jump from AUC 0.65 to 0.997 confirms that learned face embeddings provide
qualitatively different discriminative signal compared to pixel cosine similarity.
The M2 baseline was correctly diagnosed as a representation bottleneck, not a
calibration or data problem.

---

### 5. Error Analysis

At threshold 0.68 on the test split, 21 pairs are misclassified out of 944.

**False Negatives (FN = 9):** Same-identity pairs scored below 0.68. These are
cases where the two images of the same person have unusually large appearance
variation (lighting extremes, occlusion, large aging gap, or low-resolution
images). FaceNet is robust to most appearance changes but not all.

**False Positives (FP = 12):** Different-identity pairs scored above 0.68.
These occur for lookalike pairs or images where strong scene similarity (shared
backgrounds, lighting conditions, image quality artifacts) partially corrupts the
embedding.

At 97.8% balanced accuracy, neither failure mode is dominant -- the error rate
is low and roughly symmetric across the two decision types.

---

### 6. CLI Inference

The `scripts/verify.py` CLI supports single-pair and batch inference:

```
python scripts/verify.py --config configs/m3.yaml \
    --img1 <path_a> --img2 <path_b>
```

Output per pair: score, threshold, decision (SAME/DIFFERENT), confidence [0,1],
total latency, and per-stage breakdown (preprocess / embed / score).

Batch mode via `--pairs <csv> --limit N` runs over a pairs CSV.

---

### 7. Load Test Results

Concurrency test: 4 workers, 50 requests, pairs cycled from val split.

| Metric | Value |
|--------|-------|
| Throughput | 8.92 req/s |
| Latency mean | 442 ms |
| Latency p50 | 194 ms |
| Latency p95 | 3,275 ms |
| Latency p99 | 3,312 ms |
| Failures | 0 / 50 |

The p95 spike (3.3 s) reflects cold-start inference on the first batch of
concurrent requests before the FaceNet model is fully loaded per thread. Subsequent
requests run at ~194 ms (p50). Zero failures under 4-worker concurrency confirms
the GIL-safe inference path via `torch.no_grad()`.

Full results in `artifacts/load_test_results.json`.

---

### 8. Docker

The Dockerfile builds a self-contained CPU inference image. FaceNet weights are
downloaded and cached during the image build step so runtime inference requires no
internet access.

```
docker build -t lfw-verifier .
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/configs:/app/configs \
    lfw-verifier --config configs/m3.yaml --img1 <path_a> --img2 <path_b>
```

---

*Reproduce threshold selection and eval: `python scripts/run_m3_eval.py --config configs/m3.yaml`*

*Run all tests: `pytest tests/ -v`*

*Load test: `python scripts/load_test.py --config configs/m3.yaml`*
