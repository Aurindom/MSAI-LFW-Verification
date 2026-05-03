# LFW Face Verification — Final Release (v1.0-final)

Binary face verification on the **Labeled Faces in the Wild (LFW)** dataset using
FaceNet (InceptionResnetV1, VGGFace2 pretrained) embeddings. Given two face images,
the system produces a cosine-similarity score from 512-d face embeddings and makes a
same-person / different-person decision using a calibrated threshold.

---

## Quick Start

No data download required for tests and Docker inference — sample images are included in the repo.

**Install dependencies**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install facenet-pytorch --no-deps
pip install -r requirements.txt
```

**Run all tests**
```bash
pytest tests/ -v
```

**Build Docker image**
```bash
docker build -t lfw-verifier .
```

**Docker inference — Linux / Mac / Git Bash**
```bash
# smoke test
docker run --rm lfw-verifier --help

# single pair
docker run --rm \
    -v $(pwd)/samples:/app/samples \
    -v $(pwd)/configs:/app/configs \
    lfw-verifier --config configs/m3.yaml \
    --img1 samples/Aaron_Peirsol/Aaron_Peirsol_0001.jpg \
    --img2 samples/Aaron_Peirsol/Aaron_Peirsol_0002.jpg

# batch
docker run --rm \
    -v $(pwd)/samples:/app/samples \
    -v $(pwd)/configs:/app/configs \
    lfw-verifier --config configs/m3.yaml \
    --pairs samples/pairs.csv
```

**Docker inference — Windows PowerShell**
```powershell
# smoke test
docker run --rm lfw-verifier --help

# single pair
docker run --rm `
    -v "${PWD}/samples:/app/samples" `
    -v "${PWD}/configs:/app/configs" `
    lfw-verifier --config configs/m3.yaml `
    --img1 samples/Aaron_Peirsol/Aaron_Peirsol_0001.jpg `
    --img2 samples/Aaron_Peirsol/Aaron_Peirsol_0002.jpg

# batch
docker run --rm `
    -v "${PWD}/samples:/app/samples" `
    -v "${PWD}/configs:/app/configs" `
    lfw-verifier --config configs/m3.yaml `
    --pairs samples/pairs.csv
```

---

## Project Overview

| Milestone | What was built |
|-----------|---------------|
| **M1** | Real LFW ingestion via `sklearn.datasets.fetch_lfw_people`, deterministic 70/15/15 identity-disjoint split (seed=42), image file saving, pair CSV generation from actual file paths. |
| **M2** | Pixel cosine similarity baseline, threshold calibration (val sweep, rule-based selection), 5 tracked experiments, error analysis, data-centric improvement (balance enforcement + identity cap), pipeline validation, unit and integration tests. |
| **M3** | FaceNet embedding stage (InceptionResnetV1, VGGFace2), calibrated threshold (AUC=0.999, bal_acc=0.978), CLI inference with score/decision/confidence/latency, Docker packaging, concurrency load test. |
| **M4** | System Card, hardware-aware profiling (per-stage latency + batch sensitivity), reproducibility checklist, final release alignment, `v1.0-final` tag. |

---

## Pipeline Summary

```
Image A, Image B
     │
     ▼  Preprocessing
     │  Load → resize to 160×160 → normalize to [-1, 1]
     │
     ▼  Embedding generation
     │  InceptionResnetV1 (VGGFace2) → 512-d embedding
     │
     ▼  Similarity scoring
     │  Cosine similarity → mapped to [0, 1]
     │
     ▼  Threshold decision
     │  score >= 0.68 → SAME, else DIFFERENT
     │
     ▼  Confidence computation
        |score - threshold| / room on predicted side → [0, 1]
```

**Embedding model:** `InceptionResnetV1` pretrained on VGGFace2 via `facenet-pytorch`.  
512-d L2-normalised embeddings. Input: 160×160 RGB, normalised to [-1, 1].

**Threshold:** 0.68, selected on the val split (v2 pairs) by maximising balanced accuracy.  
The test split was never inspected during calibration.

---

## Final Results

| Metric | M2 Pixel Baseline | M3 FaceNet (Final) |
|--------|------------------|-------------------|
| AUC (val) | 0.649 | **0.9987** |
| Balanced Accuracy (test) | 0.618 | **0.9778** |
| F1 (test) | 0.570 | **0.9778** |
| Threshold | 0.970 | **0.68** |

**Hardware profile (CPU, Intel Core i7-10750H):**
- End-to-end latency: ~74.5 ms/pair
- Embedding stage: 72.5 ms (97.3% of total)
- Throughput: ~13 pairs/sec (sequential CPU inference)

**Load test (4 workers, 50 requests):** 8.92 req/s, p50=194 ms, p95=3,275 ms, 0 failures.

---

## Final Release Artifacts

| Artifact | Location |
|---------|---------|
| System Card | `reports/system_card.md` |
| Profiling report | `reports/profiling_report.md` |
| Reproducibility checklist | `reports/reproducibility_checklist.md` |
| M3 evaluation report | `reports/milestone3_report.md` |
| M2 evaluation report | `reports/milestone2_report.md` |
| Final config | `configs/m3.yaml` |
| Profiling raw results | `artifacts/profiling_results.json` |
| Run summaries | `artifacts/runs_summary.csv` |
| Run details | `artifacts/runs/` |
| Load test results | `artifacts/load_test_results.json` |

---

## Repository Structure

```
lfw-verification/
├── samples/                     # Committed sample images for Docker testing
│   ├── Aaron_Peirsol/
│   ├── Adam_Sandler/
│   └── pairs.csv
├── configs/
│   ├── m1.yaml                  # Ingestion config
│   ├── m2.yaml                  # M2 pixel-baseline evaluation config
│   └── m3.yaml                  # Final inference config (threshold=0.68)
├── scripts/
│   ├── ingest_lfw.py            # Downloads LFW, saves images, writes manifest
│   ├── make_pairs.py            # Pair CSV generation (v1 & v2 policies)
│   ├── run_all_experiments.py   # 5 M2 tracked experiments
│   ├── run_m3_eval.py           # M3 threshold re-selection + eval reporting
│   ├── verify.py                # CLI inference (single pair or batch)
│   ├── load_test.py             # Concurrency load test
│   └── profile_system.py        # Hardware profiling (per-stage latency + batch sensitivity)
├── src/
│   ├── ingestion.py             # LFW download, image save, split map
│   ├── pairs.py                 # Identity pool builder, pair generators
│   ├── embeddings.py            # FaceNet preprocessing + embedding extraction
│   ├── inference.py             # Per-pair inference with stage-level latency tracking
│   ├── scoring.py               # M2 pixel cosine similarity scorer
│   ├── evaluation.py            # Metrics, threshold sweep, ROC, AUC
│   ├── tracking.py              # Run logging (JSON + CSV)
│   └── validation.py            # Fail-fast pipeline checks
├── tests/
│   ├── test_inference.py        # Unit tests for embeddings + inference
│   ├── test_metrics.py          # Unit tests for evaluation functions
│   ├── test_validation.py       # Unit tests for validation checks
│   └── test_integration.py      # End-to-end pipeline integration tests
├── reports/
│   ├── system_card.md           # Final system card (M4)
│   ├── profiling_report.md      # Hardware profiling report (M4)
│   ├── reproducibility_checklist.md  # Reproducibility checklist (M4)
│   ├── milestone3_report.md     # M3 evaluation report
│   └── milestone2_report.md     # M2 evaluation report
├── artifacts/
│   ├── profiling_results.json   # Raw profiling output
│   ├── runs/                    # Tracked run JSON files
│   ├── runs_summary.csv         # Run summary table
│   └── load_test_results.json   # Load test results
├── Makefile
├── Dockerfile
└── requirements.txt
```

---

## Make Targets

> `make` requires GNU Make. On Windows, use Git Bash or run the commands directly.

| Command | What it does |
|---------|-------------|
| `make install` | Install all Python dependencies |
| `make test` | Run all tests with pytest |
| `make docker-build` | Build the Docker image |
| `make docker-test` | Build + run all Docker tests using sample images |
| `make setup-data` | Download LFW + generate pair CSVs |

---

## Full Evaluation (optional)

Only needed to reproduce the M3 threshold selection and test metrics.
Requires ~200 MB download and takes several minutes to embed all pairs.

**Linux / Mac / Git Bash**
```bash
make setup-data
python scripts/run_m3_eval.py --config configs/m3.yaml
python scripts/load_test.py --config configs/m3.yaml
```

**Windows PowerShell**
```powershell
python scripts/ingest_lfw.py --config configs/m1.yaml
python scripts/make_pairs.py --config configs/m2.yaml --version v1
python scripts/make_pairs.py --config configs/m2.yaml --version v2
python scripts/run_m3_eval.py --config configs/m3.yaml
python scripts/load_test.py --config configs/m3.yaml
```

Expected output: AUC ~0.999, balanced accuracy ~0.978, matching `reports/milestone3_report.md`.

---

## Hardware Profiling

```bash
python scripts/profile_system.py --config configs/m3.yaml --output artifacts/profiling_results.json
```

Expected: per-stage latency breakdown (embedding ~72 ms, 97% of total), batch-size sensitivity table.
Full analysis in `reports/profiling_report.md`.

---

## Using Your Own Images

Any two JPEG face images work. Replace the sample paths with your own:

**Linux / Mac / Git Bash**
```bash
docker run --rm \
    -v $(pwd)/your_images:/app/your_images \
    -v $(pwd)/configs:/app/configs \
    lfw-verifier --config configs/m3.yaml \
    --img1 your_images/person_a.jpg \
    --img2 your_images/person_b.jpg
```

**Windows PowerShell**
```powershell
docker run --rm `
    -v "${PWD}/your_images:/app/your_images" `
    -v "${PWD}/configs:/app/configs" `
    lfw-verifier --config configs/m3.yaml `
    --img1 your_images/person_a.jpg `
    --img2 your_images/person_b.jpg
```

---

## Git Tags

| Tag | Description |
|-----|------------|
| `v0.1` | Milestone 1 complete |
| `v0.2` | Milestone 2 complete |
| `v0.3` | Milestone 3 complete |
| `v1.0-final` | Final release — System Card, profiling, reproducibility checklist |
