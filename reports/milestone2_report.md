# Milestone 2 Evaluation Report
## LFW Face Verification -- Iterative ML System

---

### 1. System Overview

This report documents the Milestone 2 evaluation of a face verification pipeline
built on the Labeled Faces in the Wild (LFW) dataset. The system ingests real LFW
images via `sklearn.datasets.fetch_lfw_people`, applies a deterministic
identity-disjoint 70/15/15 train/val/test split, generates pair CSVs from actual
image file paths, computes pixel-level cosine similarity scores, and makes a binary
same-person / different-person decision using a calibrated threshold.

**Scoring method (M2 baseline):** Each image is resized to 64x64, flattened to a
12,288-dimensional vector, L2-normalised, and compared via cosine similarity. The
raw cosine value is mapped to [0, 1] via (cosine + 1) / 2 to produce the
similarity score. Higher scores indicate more similar images. This is an honest
weak pixel baseline -- it captures global colour and texture similarity but not
identity-discriminative features, making it an appropriate starting point before
upgrading to learned embeddings in Milestone 3.

**Dataset:** 1,680 real LFW identities (min 2 images each), 9,164 total images.
Split: train=1,176 identities / val=252 / test=252 (seed=42, deterministic).

---

### 2. Baseline Setup (Data Version v1)

**Pair data (v1):** Generated from real image file paths. Each identity contributes
up to 3 positive pairs (same-identity, consecutive image samples) and 3 negative
pairs (cross-identity samples). No identity cap applied. Splits: val=1,240 pairs
(pos=484, neg=756), test=1,228 pairs (pos=472, neg=756).

**Threshold rule:** The operating threshold is chosen on the **validation split**
by maximising balanced accuracy = (TPR + TNR) / 2. The test split is never
inspected during calibration.

**Tracked runs:**

| Run | Purpose | Data | Split | Threshold | Bal. Acc. |
|-----|---------|------|-------|-----------|-----------|
| run_001 | Baseline threshold sweep | v1 | val | 0.960 (auto) | 0.5966 |
| run_002 | Baseline selected-threshold eval | v1 | val | 0.960 | 0.5966 |
| run_003 | Baseline final reporting | v1 | test | 0.960 | 0.5935 |
| run_004 | Post-change threshold sweep | v2 | val | 0.970 (auto) | 0.6064 |
| run_005 | Post-change final reporting | v2 | test | 0.970 | 0.6176 |

Full details in `artifacts/runs_summary.csv`. Reproduce with:

```
python scripts/run_all_experiments.py --config configs/m2.yaml
```

---

### 3. ROC Curve and Threshold Selection

**Figure 1** (`reports/figures/roc_v1_val.png`) shows the ROC curve for the
baseline pixel-cosine verifier on the validation split.

- **AUC: 0.637** -- consistent with a weak pixel-level baseline. Pixel cosine
  similarity captures some global similarity signal but is far from the
  discriminative power of a trained face embedding. This is expected and
  motivates the embedding upgrade in Milestone 3.
- The balanced-accuracy sweep peaks at **threshold = 0.960**, which becomes the
  locked operating threshold for baseline evaluation.

The high threshold value (0.960) reflects the score distribution of pixel cosine
similarity on face images: since all images share roughly similar skin tones and
head shapes, raw cosine similarity scores cluster near the high end of [0, 1].
The threshold is calibrated on val and applied to test without modification.

**Figure 2** (`reports/figures/cm_v1_test.png`) shows the confusion matrix at
threshold 0.960 on the test split (Run 3):

| | Pred: Different | Pred: Same |
|---|---|---|
| **True: Different** | TN = 356 | FP = 400 |
| **True: Same** | FN = 134 | TP = 338 |

- **TPR (recall):** 338 / (338+134) = 0.716 -- genuine pairs correctly accepted
- **TNR (specificity):** 356 / (356+400) = 0.471 -- impostors correctly rejected
- **Balanced Accuracy:** 0.594
- **F1:** 0.559

Val-to-test transfer: val bal_acc = 0.5966, test = 0.5935, delta = -0.003.
The threshold generalises stably -- the small gap shows the calibration is not
overfit to the validation split.

---

### 4. Data-Centric Improvement (v1 to v2)

**Problem identified in v1:** With 3 negative pairs per identity and no cap, the
v1 set contains significantly more negative than positive pairs (756 vs 484 on val).
This imbalance means balanced-accuracy optimisation must work against a skewed
label distribution.

**Changes made in v2:**
1. **Balance enforcement** -- positive and negative pairs are trimmed to equal
   counts (n_pos == n_neg), removing majority-class bias.
2. **Identity cap** -- each identity contributes at most 6 pairs (3 positive + 3
   negative via budget), preventing over-representation of frequently photographed
   subjects.
3. **Duplicate filter** -- pairs where left_path == right_path are rejected.

V2 produces val=968 pairs (484 pos, 484 neg) and test=944 pairs (472 pos, 472 neg).

**Before vs. after (test split):**

| Metric | v1 (baseline) | v2 (improved) | Change |
|--------|---------------|---------------|--------|
| Balanced Accuracy | 0.5935 | 0.6176 | +0.0241 |
| F1 | 0.5587 | 0.5697 | +0.0110 |
| ROC AUC (val) | 0.6369 | 0.6494 | +0.0125 |
| Threshold | 0.960 | 0.970 | +0.010 |

**Interpretation:** Enforcing label balance in v2 gives the threshold calibration
a cleaner optimisation signal. The selected threshold shifts from 0.960 to 0.970,
reflecting the rebalanced score distribution. On the balanced v2 test set,
balanced accuracy improves by +2.4 pp and F1 by +1.1 pp. The data-centric
change had a clear positive effect.

---

### 5. Error Analysis

#### Slice 1 -- Boundary-Region Pairs (Ambiguous Scores)

**Definition:** Test pairs from the v1 set where the similarity score falls within
0.05 of the selected threshold 0.960, i.e., scores in [0.910, 1.000].

**Failure pattern:** The pixel cosine scores for both same- and different-identity
pairs overlap heavily near the high end of the [0, 1] range because all face images
share broad visual structure. Near the threshold, the scorer cannot separate genuine
pairs from impostors using pixel information alone. Both classes cluster in the
[0.90, 1.00] range, making any fixed threshold unreliable for borderline pairs.

**Hypothesis:** The score distribution overlap near 0.96 is an inherent limitation
of the pixel-level representation. No threshold can cleanly separate the classes
in this region without additional discriminative signal.

**Future fix:** Replace pixel cosine similarity with a trained face embedding
(Milestone 3). Learned embeddings push same-identity scores toward 1.0 and
different-identity scores toward 0.0, spreading the distributions apart and
dramatically reducing the boundary ambiguity zone.

---

#### Slice 2 -- False Negatives: Same-Identity Pairs Incorrectly Rejected

**Definition:** v1 test pairs with ground-truth label = 1 (same identity) where
the predicted label is 0 (different identity). Run 3: FN = 134 of 472 positive
pairs = 28.4% miss rate.

**Failure pattern:** Same-identity pairs receive scores below 0.960 when the two
images have significant appearance variation -- different lighting, expressions,
aging, or image quality. The pixel cosine score is highly sensitive to these
surface-level changes because it treats all pixel channels equally without any
identity-discriminative weighting.

**Hypothesis:** The high FN rate (28%) is a direct consequence of the weak pixel
baseline. Same-identity pairs with non-trivial appearance changes fall below the
threshold because the pixel vectors diverge even when the identity is the same.
Face-oriented embeddings learn to be robust to exactly these appearance variations,
which is why the embedding upgrade in Milestone 3 is critical.

**Future fix:** Train or load a face embedding model that maps appearance-varying
images of the same person to nearby vectors in embedding space, while mapping
different-identity images far apart. This directly addresses the root cause of the
high FN rate.

---

### 6. Lessons Learned

The most important lesson from this iteration is that **the scoring representation
is the primary bottleneck, not the threshold or the data split.** A pixel cosine
baseline produces AUC ~0.64 -- better than random (0.50) but far from practical
(0.95+). The threshold calibration discipline (val sweep, locked test reporting)
transferred well with only -0.003 delta, confirming the evaluation protocol is
sound.

The data-centric v2 improvement validated that **label balance materially affects
calibration quality**: enforcing equal positive and negative counts improved both
balanced accuracy (+2.4 pp) and F1 (+1.1 pp), and the AUC improvement (+0.013)
suggests the balanced set gives a cleaner view of true discriminability.

The remaining high error rates are an inherent consequence of pixel-level
scoring and will be directly addressed by the embedding upgrade in Milestone 3.

---

*Figures: `reports/figures/roc_v1_val.png`, `roc_v2_val.png`, `cm_v1_test.png`, `cm_v2_test.png`*

*Reproduce all runs: `python scripts/run_all_experiments.py --config configs/m2.yaml`*

*Run tests: `pytest tests/ -v`*
