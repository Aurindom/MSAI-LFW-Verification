# Reproducibility Checklist — LFW Face Verification System
**Final tag:** `v1.0-final`  
**Config:** `configs/m3.yaml`

A grader can follow these steps from a fresh clone to reproduce the core results and run the Dockerized CLI.

---

## Environment Setup

```bash
# Clone the repository
git clone https://github.com/Aurindom/MSAI-LFW-Verification.git
cd MSAI-LFW-Verification
git checkout v1.0-final

# Install dependencies (CPU PyTorch + FaceNet)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install facenet-pytorch --no-deps
pip install -r requirements.txt
```

**Expected:** All packages install without errors. `facenet-pytorch --no-deps` is required to avoid a numpy version conflict with numpy 2.x.

---

## Run All Tests

```bash
pytest tests/ -v
```

**Expected:** All tests pass. No data download required — tests use synthetic data and sample images.

---

## Docker — Build and Smoke Test

```bash
# Build image (FaceNet weights downloaded and cached at build time)
docker build -t lfw-verifier .

# Smoke test
docker run --rm lfw-verifier --help
```

**Expected:** Image builds successfully (~2–5 min first time, weights ~90 MB). Help output prints to console.

---

## Docker — Single Pair Inference

**Linux / Mac / Git Bash:**
```bash
docker run --rm \
    -v $(pwd)/samples:/app/samples \
    -v $(pwd)/configs:/app/configs \
    lfw-verifier --config configs/m3.yaml \
    --img1 samples/Aaron_Peirsol/Aaron_Peirsol_0001.jpg \
    --img2 samples/Aaron_Peirsol/Aaron_Peirsol_0002.jpg
```

**Windows PowerShell:**
```powershell
docker run --rm `
    -v "${PWD}/samples:/app/samples" `
    -v "${PWD}/configs:/app/configs" `
    lfw-verifier --config configs/m3.yaml `
    --img1 samples/Aaron_Peirsol/Aaron_Peirsol_0001.jpg `
    --img2 samples/Aaron_Peirsol/Aaron_Peirsol_0002.jpg
```

**Expected output (approximate):**
```
Decision:   SAME
Score:      ~0.85–0.95
Threshold:  0.6800
Confidence: ~0.7–0.9
```

---

## Docker — Batch Inference

**Linux / Mac / Git Bash:**
```bash
docker run --rm \
    -v $(pwd)/samples:/app/samples \
    -v $(pwd)/configs:/app/configs \
    lfw-verifier --config configs/m3.yaml \
    --pairs samples/pairs.csv
```

**Windows PowerShell:**
```powershell
docker run --rm `
    -v "${PWD}/samples:/app/samples" `
    -v "${PWD}/configs:/app/configs" `
    lfw-verifier --config configs/m3.yaml `
    --pairs samples/pairs.csv
```

**Expected:** 4 pairs processed, 2 SAME and 2 DIFFERENT decisions. No failures.

---

## Reproduce M3 Threshold Selection and Test Metrics

Requires the full LFW dataset (~200 MB download). This step is optional for grading — results are already reported in `reports/milestone3_report.md`.

```bash
# Step 1 — Download LFW and save images to disk
python scripts/ingest_lfw.py --config configs/m1.yaml

# Step 2 — Generate pair CSVs (v1 and v2 policies)
python scripts/make_pairs.py --config configs/m2.yaml --version v1
python scripts/make_pairs.py --config configs/m2.yaml --version v2

# Step 3 — Run M3 threshold re-selection and test evaluation
python scripts/run_m3_eval.py --config configs/m3.yaml
```

**Expected output:** AUC ~0.999, balanced accuracy ~0.978 on test split, threshold confirmed at 0.68. Matches `reports/milestone3_report.md`.

---

## Run Hardware Profiling

```bash
python scripts/profile_system.py --config configs/m3.yaml --output artifacts/profiling_results.json
```

**Expected:** Per-stage latency table printed to console (embedding ~72 ms, ~97% of end-to-end). JSON results saved to `artifacts/profiling_results.json`. Matches `reports/profiling_report.md`.

---

## Key Artifact Locations

| Artifact | Path |
|---------|------|
| Final config | `configs/m3.yaml` |
| System Card | `reports/system_card.md` |
| Profiling report | `reports/profiling_report.md` |
| M3 evaluation report | `reports/milestone3_report.md` |
| Profiling raw results (JSON) | `artifacts/profiling_results.json` |
| Run summaries (CSV) | `artifacts/runs_summary.csv` |
| Run details (JSON) | `artifacts/runs/` |
| Sample images | `samples/` |
| Docker entrypoint | `scripts/verify.py` |
| Profiling script | `scripts/profile_system.py` |
