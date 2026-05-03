# System Card — LFW Face Verification System
**Version:** v1.0-final  
**Date:** 2026-05-03  
**Course:** MSML/MSAI 605

---

## 1. System Overview

This system performs binary face verification on pairs of face images. Given two images, it determines whether they depict the same person or different people, producing a similarity score, a binary decision, and a calibrated confidence value.

**Pipeline:**

```
Image A, Image B
     │
     ▼  Preprocessing
     │  Load → resize to 160×160 RGB → normalise to [-1, 1]
     │
     ▼  Embedding generation
     │  InceptionResnetV1 (VGGFace2 pretrained) → 512-d L2-normalised embedding
     │
     ▼  Similarity scoring
     │  Cosine similarity → mapped to [0, 1] via (cosine + 1) / 2
     │
     ▼  Threshold decision
        score ≥ 0.68 → SAME PERSON, else DIFFERENT PERSON
```

**Model:** `InceptionResnetV1` pretrained on VGGFace2 via `facenet-pytorch 2.6.0`.  
**Embedding:** 512-dimensional, L2-normalised.  
**Input:** 160×160 RGB JPEG images, normalised to [-1, 1].  
**Threshold:** 0.68, selected on the validation split by maximising balanced accuracy (Youden-J rule). The test split was never inspected during calibration.

---

## 2. Intended Use

**Supported uses:**
- Academic benchmarking and research on the LFW dataset
- Offline face verification in controlled, low-stakes settings
- Demonstration of embedding-based similarity pipelines
- Evaluation of pretrained face representation models

**Out-of-scope uses:**
- Production identity verification or authentication systems
- Law enforcement, surveillance, or security screening
- Any safety-critical or legally sensitive decision-making
- Real-time or high-throughput production deployment without further validation
- Use on populations or image conditions not represented in LFW

---

## 3. Data Summary

**Dataset:** Labeled Faces in the Wild (LFW), fetched via `sklearn.datasets.fetch_lfw_people`.  
**Total:** 1,680 identities, 9,164 images.  
**Split (seed=42, identity-disjoint):**

| Split | Identities | Pairs (v2) |
|-------|-----------|-----------|
| Train | 1,176 (70%) | — |
| Val   | 252 (15%) | 968 (484 pos, 484 neg) |
| Test  | 252 (15%) | 944 (472 pos, 472 neg) |

**Pair policy (v2):** Balance enforcement (n_pos == n_neg), per-identity cap of 6 pairs, duplicate filtering. Identities are fully disjoint across splits — no identity appears in more than one split.

**Data limitations:**
- LFW is heavily skewed toward public figures, predominantly from Western and English-speaking media contexts.
- The dataset has known demographic imbalances: it over-represents male subjects and public figures from specific cultural backgrounds.
- Image quality, resolution, and pose vary significantly across identities — some identities have only 2 images, limiting intra-identity diversity.
- The dataset does not include verified demographic metadata (age, gender, ethnicity), making formal subgroup analysis unreliable.

---

## 4. Operating Threshold and Key Metrics

**Threshold:** 0.68  
**Selection rule:** Maximum balanced accuracy on the validation split (v2 pairs).  
**Calibration split:** Val only. Test split was held out until final evaluation.

**Validation metrics at threshold 0.68:**

| Metric | Value |
|--------|-------|
| AUC | 0.9987 |
| Balanced Accuracy | 0.9835 |
| F1 | 0.9835 |
| TPR (Sensitivity) | 0.9855 |
| FPR (1 - Specificity) | 0.0186 |

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
| Total pairs | 944 |

Val-to-test balanced accuracy delta: −0.0057. The threshold generalises cleanly with no evidence of val overfitting.

**M2 vs M3 comparison:**

| Metric | M2 Pixel Baseline | M3 FaceNet |
|--------|------------------|------------|
| AUC (val) | 0.649 | **0.9987** |
| Balanced Accuracy (test) | 0.618 | **0.9778** |
| F1 (test) | 0.570 | **0.9778** |
| Threshold | 0.970 | 0.68 |

---

## 5. Failure Modes and Limitations

**False Negatives (same-person pairs scored below threshold):** 9 out of 472 same-person test pairs were missed. These occur when intra-identity appearance variation is extreme: severe lighting differences, heavy occlusion (sunglasses, hats, scarves), large aging gaps between images, or very low image resolution. FaceNet embeddings are robust to moderate appearance change but not to all combinations of these factors.

**False Positives (different-person pairs scored above threshold):** 12 out of 472 different-person test pairs were incorrectly matched. These arise for visually similar individuals (lookalikes), pairs where shared scene context (backgrounds, lighting conditions, image compression artifacts) partially aligns embeddings, or cases where image quality is poor enough to distort the embedding.

**General limitations:**
- Performance degrades for low-resolution or heavily compressed images.
- The system has no face detection step. It assumes the input image is already cropped to contain a single face. Non-face images, multi-face images, or poorly cropped images will produce unreliable embeddings and scores.
- The 0.68 threshold was calibrated on LFW v2 pairs. Performance on other datasets or real-world image distributions may differ significantly.
- Sequential CPU inference produces ~74 ms per pair. The system is not designed for high-throughput production workloads without batching or GPU acceleration.

---

## 6. Fairness-Related Risks and Misuse Concerns

**Demographic representation:** LFW is known to have demographic imbalances. Public figures from specific backgrounds are over-represented. Because the dataset does not include verified demographic metadata, we cannot report per-subgroup accuracy, and we do not make unsupported fairness claims.

**Known risk categories:**
- Face verification systems generally exhibit performance disparities across demographic groups, often with higher error rates for women, older individuals, and individuals from underrepresented ethnic backgrounds. This system has not been audited for such disparities, and such disparities may exist.
- Image quality disparities across populations (lower-quality historical images of underrepresented groups) may amplify performance differences.
- Lookalike false positives may be more frequent for individuals with similar demographic characteristics.

**Misuse risks:**
- This system must not be used for surveillance, law enforcement, or any application where false positives or false negatives carry serious consequences.
- Even at 97.8% balanced accuracy, at scale the absolute number of errors becomes large. At 1 million pairs, approximately 22,000 pairs would be misclassified.
- The system outputs a continuous confidence score. Over-reliance on high-confidence decisions without human review is inadvisable in any sensitive context.

---

## 7. Operational Constraints

| Constraint | Detail |
|-----------|--------|
| Hardware | CPU-only. No GPU required. Tested on Intel Core i7-10750H (12 logical cores). |
| OS | Tested on Windows 11. Docker image based on `python:3.11-slim`. |
| Python | 3.10+ required. Tested with 3.10.19 (local) and 3.11 (Docker). |
| Input format | JPEG images, single face per image, cropped. 160×160 resize applied internally. |
| Latency | ~74.5 ms per pair on CPU (embedding stage dominates at 97.3%). |
| Throughput | ~13.3 pairs/sec sequential CPU inference. |
| Internet | Required at build time to download FaceNet weights. Weights are cached in Docker image. Runtime inference requires no internet. |
| Memory | Model footprint ~89 MB. No GPU VRAM required. |
| Threshold | Fixed at 0.68. Override available via CLI `--threshold` flag. |

---

## 8. Reproducibility Pointer

- **Repository:** [MSAI-LFW-Verification](https://github.com/Aurindom/MSAI-LFW-Verification)
- **Final tag:** `v1.0-final`
- **Config:** `configs/m3.yaml` (threshold, model, inference settings)
- **Reproducibility checklist:** `reports/reproducibility_checklist.md`
- **Profiling report:** `reports/profiling_report.md`
- **M3 evaluation report:** `reports/milestone3_report.md`
- **README:** Entry point for environment setup, Docker commands, and CLI usage
