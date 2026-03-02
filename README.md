# LFW Face Verification — Milestone 1

A reproducible pipeline for face verification using the Labeled Faces in the Wild (LFW) dataset. Covers dataset ingestion, deterministic pair generation, and vectorized similarity scoring.

## Project Structure

```
lfw-verification/
├── src/               # Core library modules
│   ├── ingest.py      # LFW download and manifest generation
│   ├── pairs.py       # Pair generation and train/test splitting
│   └── similarity.py  # Cosine and Euclidean similarity (loop + vectorized)
├── scripts/           # Runnable entry points
│   ├── run_ingestion.py
│   ├── run_pairs.py
│   └── run_benchmark.py
├── configs/
│   └── config.yaml    # All hyperparameters and paths
├── tests/
│   └── test_similarity.py
└── requirements.txt
```

## How to Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Ingest LFW dataset**
Downloads LFW, saves `images.npy`, `labels.npy`, `names.npy`, and `manifest.json` to `data/processed/`.
```bash
python scripts/run_ingestion.py
```

**3. Generate verification pairs**
Produces deterministic positive/negative pair splits saved to `data/pairs/`.
```bash
python scripts/run_pairs.py
```

**4. Run similarity benchmark**
Times Python loop vs NumPy vectorized for cosine and Euclidean similarity, then verifies correctness.
```bash
python scripts/run_benchmark.py
```

**5. Run tests**
```bash
pytest tests/
```

## Configuration

All parameters live in `configs/config.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `data.min_faces_per_person` | 20 | Minimum images per identity |
| `data.seed` | 42 | Random seed for reproducibility |
| `data.test_size` | 0.2 | Fraction held out for testing |
| `pairs.n_positive` | 500 | Same-identity pairs |
| `pairs.n_negative` | 500 | Different-identity pairs |
| `pairs.seed` | 42 | Seed for pair sampling |

## Outputs

| File | Description |
|------|-------------|
| `data/processed/manifest.json` | Dataset statistics and split policy |
| `data/pairs/pairs_meta.json` | Pair split counts and seeds |
| `data/pairs/train_pairs.npy` | Training pair indices |
| `data/pairs/test_pairs.npy` | Test pair indices |
