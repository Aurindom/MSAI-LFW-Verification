# Profiling Report — LFW Face Verification System
**Version:** v1.0-final  
**Date:** 2026-05-03

---

## 1. Measurement Environment

| Property | Value |
|----------|-------|
| OS | Windows 11 (10.0.26200) |
| Processor | Intel64 Family 6 Model 165 Stepping 2 (Intel Core i7-10750H) |
| Logical CPU cores | 12 |
| Python | 3.10.19 (conda datasci environment) |
| PyTorch | CPU-only build |
| facenet-pytorch | 2.6.0 |
| Model | InceptionResnetV1, pretrained on VGGFace2 |
| Input | 160×160 RGB JPEG, normalised to [-1, 1] |
| Threshold | 0.68 |

**Methodology:** All timings use `time.perf_counter()` for high-resolution wall-clock measurement. Each stage is timed individually inside `src/inference.verify_pair()`. For per-stage profiling, 2 warmup runs were discarded before 5 timed runs. For batch-size sensitivity, 2 warmup batches were discarded before 5 timed batches per size. Reported values are means over the 5 timed runs.

---

## 2. Per-Stage Latency (Single Pair, CPU Baseline)

Measured on a single pair with 2 warmup + 5 timed runs.

| Stage | Mean (ms) | % of End-to-End |
|-------|-----------|----------------|
| Preprocessing | 1.966 | 2.6% |
| Embedding generation | 72.513 | 97.3% |
| Similarity scoring | 0.063 | 0.1% |
| **End-to-end** | **74.546** | 100% |

**Interpretation:** Embedding generation completely dominates latency at 97.3%. This is expected — InceptionResnetV1 is a deep convolutional network requiring a full forward pass per image (two passes per pair). Preprocessing (image load, resize, normalise) is a distant second at 2.6%. Cosine similarity on two 512-d vectors is negligible at 0.1%.

Any effort to reduce inference latency must target the embedding stage. Options include batching multiple images in a single forward pass (requires refactoring the current per-pair pipeline), GPU acceleration, or model distillation.

---

## 3. Batch-Size Sensitivity (CPU Baseline)

Batch size here means the number of pairs processed sequentially in one timed window. The current pipeline calls `verify_pair()` once per pair. Pairs within a batch are processed one at a time on CPU.

| Batch Size | Total (ms) | Per-pair (ms) | Throughput (pairs/s) |
|-----------|-----------|--------------|---------------------|
| 1 | 74.3 | 74.3 | 13.5 |
| 2 | 152.4 | 76.2 | 13.1 |
| 4 | 298.0 | 74.5 | 13.4 |
| 8 | 603.5 | 75.4 | 13.3 |
| 16 | 1,212.6 | 75.8 | 13.2 |
| 32 | 2,452.4 | 76.6 | 13.1 |

**Interpretation:** Per-pair latency and throughput are stable across all batch sizes (~74–77 ms/pair, ~13 pairs/sec). This is expected behaviour for sequential CPU inference — the model processes one pair at a time with no vectorisation benefit from grouping. Total latency scales linearly with batch size. There is no "sweet spot" batch size for this configuration.

To observe meaningful batch-size effects, the embedding step would need to be refactored to accept a batch of images in a single forward pass through the network, which would allow CPU (and especially GPU) parallelism. The current architecture prioritises simplicity and CLI usability over throughput.

---

## 4. Load Test Reference (from Milestone 3)

Concurrency test run with 4 workers and 50 requests, pairs cycled from the validation split.

| Metric | Value |
|--------|-------|
| Throughput | 8.92 req/s |
| Latency mean | 442 ms |
| Latency p50 | 194 ms |
| Latency p95 | 3,275 ms |
| Latency p99 | 3,312 ms |
| Failures | 0 / 50 |

The p95 spike reflects cold-start model loading on the first batch of concurrent requests. Subsequent requests run at ~194 ms p50 once the model is resident in memory per thread. Zero failures under 4-worker concurrency confirms the GIL-safe inference path via `torch.no_grad()`.

---

## 5. GPU Comparison

No GPU comparison is included. This system targets CPU deployment. The Docker image uses a CPU-only PyTorch build. GPU results, if needed, would require rebuilding with a CUDA-enabled PyTorch wheel and a CUDA-capable host.

---

## 6. Summary

Embedding generation accounts for 97.3% of end-to-end latency. The system delivers ~13 pairs/sec on a 12-core Intel CPU with no GPU. Throughput does not improve with larger sequential batches because inference is not vectorised across pairs at the network level. The primary bottleneck for any latency or throughput improvement is the FaceNet forward pass.
