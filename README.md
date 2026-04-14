# LFW Face Verification — Milestone 3

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

**Run all tests (70 tests, no data needed)**
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
| **M2** | Pixel cosine similarity baseline, threshold calibration (val sweep, rule-based selection), 5 tracked experiments, error analysis, data-centric improvement (balance enforcement + identity cap), pipeline validation, 70 unit + integration tests. |
| **M3** | FaceNet embedding stage (InceptionResnetV1, VGGFace2), re-selected threshold (AUC=0.999, bal_acc=0.978), CLI inference with score/decision/confidence/latency, Docker packaging, concurrency load test. |

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

**Confidence formula:**
- If SAME  (score >= threshold): `confidence = (score − threshold) / (1.0 − threshold)`
- If DIFF  (score < threshold):  `confidence = (threshold − score) / threshold`
- Range [0, 1]: 0 = right at the boundary, 1 = maximum certainty

**Embedding model:** `InceptionResnetV1` pretrained on VGGFace2 via `facenet-pytorch`.
512-d L2-normalised embeddings. Input: 160×160 RGB, normalised to [-1, 1].

**Threshold:** 0.68, selected on the val split (v2 pairs) by maximising balanced accuracy.
The test split was never inspected during calibration.

---

## Repository Structure

```
lfw-verification/
├── samples/                     # Committed sample images for quick Docker testing
│   ├── Aaron_Peirsol/
│   ├── Adam_Sandler/
│   └── pairs.csv                # 4 sample pairs (2 SAME, 2 DIFFERENT)
├── configs/
│   ├── m1.yaml                  # Ingestion config
│   ├── m2.yaml                  # M2 pixel-baseline evaluation config
│   └── m3.yaml                  # M3 embedding inference config + threshold
├── scripts/
│   ├── ingest_lfw.py            # Downloads LFW, saves images, writes manifest
│   ├── make_pairs.py            # Pair CSV generation from real image paths (v1 & v2)
│   ├── run_all_experiments.py   # 5 M2 tracked experiments
│   ├── run_m3_eval.py           # M3 threshold re-selection + eval reporting
│   ├── verify.py                # CLI inference interface (single pair or batch)
│   └── load_test.py             # Concurrency load test with throughput + p95
├── src/
│   ├── ingestion.py             # LFW download, image save, split map
│   ├── pairs.py                 # Identity pool builder, pair generators
│   ├── embeddings.py            # FaceNet preprocessing + embedding extraction
│   ├── inference.py             # 6-stage pair inference (preprocess/embed/score/decide/confidence/latency)
│   ├── scoring.py               # M2 pixel cosine similarity scorer
│   ├── evaluation.py            # Metrics, threshold sweep, ROC, AUC
│   ├── tracking.py              # Run logging (JSON + CSV)
│   └── validation.py            # Fail-fast pipeline checks
├── tests/
│   ├── test_inference.py        # Unit tests for embeddings + inference + smoke test
│   ├── test_metrics.py          # Unit tests for evaluation functions
│   ├── test_validation.py       # Unit tests for validation checks
│   └── test_integration.py     # End-to-end pipeline + CSV portability tests
├── reports/
│   ├── milestone2_report.md
│   └── milestone3_report.md
├── artifacts/
│   ├── runs/                    # Tracked run JSON files
│   └── runs_summary.csv
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
| `make test` | Run all 70 tests with pytest |
| `make docker-build` | Build the Docker image |
| `make docker-test` | Build + run all three Docker tests using sample images |
| `make setup-data` | Download LFW + generate pair CSVs (needed for full eval only) |

---

## Full Evaluation (optional)

Only needed if you want to reproduce the M3 threshold selection and test metrics.
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

---

## Full End-to-End Evaluation via Docker

Runs the complete pipeline inside Docker with no local Python environment needed.
Data is downloaded into your local `data/` folder via volume mount and persists after the container exits.

**Linux / Mac / Git Bash**
```bash
# Step 1 — Download LFW dataset (~200 MB, saved to local data/)
docker run --rm --entrypoint python \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/outputs:/app/outputs \
    -v $(pwd)/configs:/app/configs \
    lfw-verifier scripts/ingest_lfw.py --config configs/m1.yaml

# Step 2 — Generate pair CSVs
docker run --rm --entrypoint python \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/outputs:/app/outputs \
    -v $(pwd)/configs:/app/configs \
    lfw-verifier scripts/make_pairs.py --config configs/m2.yaml --version v1

docker run --rm --entrypoint python \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/outputs:/app/outputs \
    -v $(pwd)/configs:/app/configs \
    lfw-verifier scripts/make_pairs.py --config configs/m2.yaml --version v2

# Step 3 — Run M3 threshold selection + test eval
docker run --rm --entrypoint python \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/outputs:/app/outputs \
    -v $(pwd)/configs:/app/configs \
    -v $(pwd)/artifacts:/app/artifacts \
    lfw-verifier scripts/run_m3_eval.py --config configs/m3.yaml
```

**Windows PowerShell**
```powershell
# Step 1 — Download LFW dataset (~200 MB, saved to local data/)
docker run --rm --entrypoint python `
    -v "${PWD}/data:/app/data" `
    -v "${PWD}/outputs:/app/outputs" `
    -v "${PWD}/configs:/app/configs" `
    lfw-verifier scripts/ingest_lfw.py --config configs/m1.yaml

# Step 2 — Generate pair CSVs
docker run --rm --entrypoint python `
    -v "${PWD}/data:/app/data" `
    -v "${PWD}/outputs:/app/outputs" `
    -v "${PWD}/configs:/app/configs" `
    lfw-verifier scripts/make_pairs.py --config configs/m2.yaml --version v1

docker run --rm --entrypoint python `
    -v "${PWD}/data:/app/data" `
    -v "${PWD}/outputs:/app/outputs" `
    -v "${PWD}/configs:/app/configs" `
    lfw-verifier scripts/make_pairs.py --config configs/m2.yaml --version v2

# Step 3 — Run M3 threshold selection + test eval
docker run --rm --entrypoint python `
    -v "${PWD}/data:/app/data" `
    -v "${PWD}/outputs:/app/outputs" `
    -v "${PWD}/configs:/app/configs" `
    -v "${PWD}/artifacts:/app/artifacts" `
    lfw-verifier scripts/run_m3_eval.py --config configs/m3.yaml
```

Expected output: AUC ~0.999, balanced accuracy ~0.978, matching the report in `reports/milestone3_report.md`.

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

## M3 Results

| Metric | M2 Pixel Baseline (v2, test) | M3 FaceNet (v2, test) |
|--------|--------|-------|
| AUC (val) | 0.649 | **0.999** |
| Balanced Accuracy | 0.618 | **0.978** |
| F1 | 0.570 | **0.978** |
| Threshold | 0.970 | 0.680 |

**Load test (4 workers, 50 requests):**
- Throughput: 8.92 req/s
- Latency mean: 442 ms  |  p50: 194 ms  |  p95: 3,275 ms  |  p99: 3,312 ms
- Failures: 0/50

---

## Artifact Locations

| Artifact | Location |
|----------|---------|
| M2 eval report | `reports/milestone2_report.md` |
| M3 eval report | `reports/milestone3_report.md` |
| Run details (JSON) | `artifacts/runs/run_*.json` |
| Run summary (CSV) | `artifacts/runs_summary.csv` |
| Load test results | `artifacts/load_test_results.json` |

---

## Git Tags

| Tag | Description |
|-----|------------|
| `v0.1` | Milestone 1 complete |
| `v0.2` | Milestone 2 complete |
| `v0.3` | Milestone 3 complete |
