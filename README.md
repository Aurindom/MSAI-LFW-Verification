# LFW Face Verification - Milestone 2

Binary face verification on the **Labeled Faces in the Wild (LFW)** dataset.
Given two face images, the system produces a cosine-similarity score and makes a
same-person / different-person decision using a calibrated threshold.

---

## Project Overview

| Milestone | What was built |
|-----------|---------------|
| **M1** | LFW ingestion (TFDS), deterministic 70/15/15 identity-disjoint split, pair CSV generation, vectorised cosine-similarity and Euclidean-distance metrics, benchmarking script. |
| **M2** | Threshold calibration (sweep → rule-based selection on val), experiment tracking (5 runs), error analysis (2 slices), data-centric improvement (identity cap + balance enforcement), pipeline validation, unit + integration tests, evaluation report. |

---

## Repository Structure

```
lfw-verification/
├── configs/
│   ├── m1.yaml              # Milestone 1 config
│   └── m2.yaml              # Milestone 2 config
├── scripts/
│   ├── ingest_lfw.py        # LFW ingestion → manifest.json
│   ├── make_pairs.py        # Pair CSV generation (v1 & v2)
│   ├── bench_similarity.py  # Vectorised vs loop benchmark
│   ├── run_eval.py          # Single tracked evaluation run
│   └── run_all_experiments.py  # Runs all 5 M2 experiments
├── src/
│   ├── similarity.py        # Cosine + Euclidean similarity
│   ├── evaluation.py        # Metrics, threshold sweep, ROC
│   ├── scoring.py           # Deterministic score generation
│   ├── tracking.py          # Run logging (JSON + CSV)
│   └── validation.py        # Fail-fast pipeline checks
├── tests/
│   ├── test_metrics.py      # Unit tests – evaluation functions
│   ├── test_validation.py   # Unit tests – validation checks
│   └── test_integration.py  # End-to-end pipeline integration test
├── reports/
│   ├── milestone2_report.md # Milestone 2 evaluation report
│   └── figures/             # ROC plots, confusion matrices
├── artifacts/
│   └── runs/                # Tracked run JSON files + runs_summary.csv
├── outputs/
│   ├── manifest.json        # Dataset split statistics
│   ├── pairs/               # v1 pair CSVs (train/val/test)
│   └── pairs_v2/            # v2 pair CSVs (data-centric improvement)
└── requirements.txt
```

---

## Environment Setup

```bash
python -m venv .venv

.venv\Scripts\activate

source .venv/bin/activate

pip install -r requirements.txt
```

---

## How to Run

### Step 1 – Generate pair CSVs

```bash

python scripts/make_pairs.py --config configs/m2.yaml --version v1


python scripts/make_pairs.py --config configs/m2.yaml --version v2
```

### Step 2 – Run all 5 tracked experiments (recommended)

```bash
python scripts/run_all_experiments.py --config configs/m2.yaml
```

This generates:
- 5 tracked run JSON files in `artifacts/runs/`
- `artifacts/runs_summary.csv` — compact comparison table
- ROC curve plots and confusion matrices in `reports/figures/`

### Step 3 – Run individual evaluations (optional)

```bash

python scripts/run_eval.py --config configs/m2.yaml --split val \
    --data-version v1 --note "baseline threshold sweep"


python scripts/run_eval.py --config configs/m2.yaml --split test \
    --data-version v1 --threshold 0.50 --note "baseline test reporting"
```

### Step 4 – Run tests

```bash
pytest tests/ -v
```

### Step 5 – Benchmark similarity metrics (Milestone 1)

```bash
python scripts/bench_similarity.py --config configs/m1.yaml
```

---

## Reproducing the Main Reported Result

```bash

git clone <repo-url>
cd lfw-verification
pip install -r requirements.txt


python scripts/make_pairs.py --config configs/m2.yaml --version v1
python scripts/make_pairs.py --config configs/m2.yaml --version v2


python scripts/run_all_experiments.py --config configs/m2.yaml


pytest tests/ -v
```

The final baseline test result (Run 3) and improved result (Run 5) are printed
to stdout and stored in `artifacts/runs_summary.csv`.

---

## Threshold Selection Rule

The operating threshold is chosen on the **validation split** by maximising
**balanced accuracy = (TPR + TNR) / 2**. This treats false accepts and false
rejects as equally costly and is computed before the test split is ever inspected.
The rule is configured under `threshold_selection.rule` in `configs/m2.yaml`.

---

## Data-Centric Improvement (v1 → v2)

| Change | Reason |
|--------|--------|
| Identity cap (max 8 pairs/identity) | Prevents high-frequency identities from dominating the metric signal |
| Balance enforcement (n_pos == n_neg) | Removes majority-class bias from accuracy-based metrics |
| Duplicate filter (left ≠ right path) | Eliminates trivially-correct self-comparison pairs |

---

## Report and Artifacts

| Artifact | Location |
|----------|---------|
| Evaluation report | `reports/milestone2_report.md` |
| ROC curves | `reports/figures/roc_*.png` |
| Confusion matrices | `reports/figures/cm_*.png` |
| Run details (JSON) | `artifacts/runs/run_*.json` |
| Run summary (CSV) | `artifacts/runs_summary.csv` |

---

## Git Tags

| Tag | Description |
|-----|------------|
| `v0.1` | Milestone 1 complete |
| `v0.2` | Milestone 2 complete |
