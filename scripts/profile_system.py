import argparse
import json
import os
import platform
import sys
import time

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference import verify_pair

SAMPLE_PAIRS = [
    ("samples/Aaron_Peirsol/Aaron_Peirsol_0001.jpg", "samples/Aaron_Peirsol/Aaron_Peirsol_0002.jpg"),
    ("samples/Adam_Sandler/Adam_Sandler_0000.jpg", "samples/Adam_Sandler/Adam_Sandler_0001.jpg"),
    ("samples/Aaron_Peirsol/Aaron_Peirsol_0001.jpg", "samples/Adam_Sandler/Adam_Sandler_0000.jpg"),
    ("samples/Aaron_Peirsol/Aaron_Peirsol_0002.jpg", "samples/Adam_Sandler/Adam_Sandler_0001.jpg"),
]

BATCH_SIZES = [1, 2, 4, 8, 16, 32]
N_WARMUP = 2
N_TIMED = 5


def system_info() -> dict:
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count_logical": os.cpu_count(),
    }


def run_pairs(pairs: list, threshold: float, image_size: int) -> list:
    return [verify_pair(a, b, threshold, image_size) for a, b in pairs]


def profile_per_stage(threshold: float, image_size: int) -> dict:
    pair = SAMPLE_PAIRS[0]

    for _ in range(N_WARMUP):
        verify_pair(pair[0], pair[1], threshold, image_size)

    preprocess_ms, embed_ms, score_ms, total_ms = [], [], [], []
    for _ in range(N_TIMED):
        r = verify_pair(pair[0], pair[1], threshold, image_size)
        preprocess_ms.append(r.preprocess_s * 1000)
        embed_ms.append(r.embed_s * 1000)
        score_ms.append(r.score_s * 1000)
        total_ms.append(r.latency_s * 1000)

    def stats(values: list) -> dict:
        mean = sum(values) / len(values)
        return {"mean_ms": round(mean, 3), "min_ms": round(min(values), 3), "max_ms": round(max(values), 3)}

    return {
        "warmup_runs": N_WARMUP,
        "timed_runs": N_TIMED,
        "preprocess": stats(preprocess_ms),
        "embed": stats(embed_ms),
        "score": stats(score_ms),
        "end_to_end": stats(total_ms),
    }


def profile_batch_sensitivity(threshold: float, image_size: int) -> list:
    results = []
    for batch_size in BATCH_SIZES:
        pairs = [SAMPLE_PAIRS[i % len(SAMPLE_PAIRS)] for i in range(batch_size)]

        for _ in range(N_WARMUP):
            run_pairs(pairs, threshold, image_size)

        run_times = []
        for _ in range(N_TIMED):
            t0 = time.perf_counter()
            run_pairs(pairs, threshold, image_size)
            run_times.append(time.perf_counter() - t0)

        mean_total_ms = sum(run_times) / len(run_times) * 1000
        mean_per_pair_ms = mean_total_ms / batch_size
        throughput = batch_size / (mean_total_ms / 1000)

        results.append({
            "batch_size": batch_size,
            "mean_total_ms": round(mean_total_ms, 3),
            "mean_per_pair_ms": round(mean_per_pair_ms, 3),
            "throughput_pairs_per_sec": round(throughput, 2),
        })
    return results


def print_summary(info: dict, per_stage: dict, batch: list) -> None:
    print("\n=== LFW Verifier — Hardware Profile ===")
    print(f"Platform  : {info['platform']}")
    print(f"Processor : {info['processor']}")
    print(f"Python    : {info['python']}")
    print(f"CPU cores : {info['cpu_count_logical']}")

    e2e = per_stage["end_to_end"]["mean_ms"]
    pre = per_stage["preprocess"]["mean_ms"]
    emb = per_stage["embed"]["mean_ms"]
    scr = per_stage["score"]["mean_ms"]

    print(f"\n--- Per-Stage Latency (single pair, {N_TIMED} timed runs after {N_WARMUP} warmups) ---")
    print(f"  Preprocessing : {pre:.3f} ms  ({pre/e2e*100:.1f}%)")
    print(f"  Embedding     : {emb:.3f} ms  ({emb/e2e*100:.1f}%)")
    print(f"  Scoring       : {scr:.3f} ms  ({scr/e2e*100:.1f}%)")
    print(f"  End-to-end    : {e2e:.3f} ms")

    print(f"\n--- Batch-Size Sensitivity ---")
    print(f"  {'Batch':>6}  {'Total (ms)':>12}  {'Per-pair (ms)':>14}  {'Throughput (pairs/s)':>20}")
    for row in batch:
        print(f"  {row['batch_size']:>6}  {row['mean_total_ms']:>12.1f}  {row['mean_per_pair_ms']:>14.1f}  {row['throughput_pairs_per_sec']:>20.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile FaceNet inference pipeline stage latency and batch sensitivity.")
    parser.add_argument("--config", required=True, help="Path to m3.yaml config.")
    parser.add_argument("--output", default="artifacts/profiling_results.json", help="Path to save JSON results.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    threshold = config["threshold"]["value"]
    image_size = config["inference"]["image_size"]

    print("Warming up model...")
    info = system_info()
    per_stage = profile_per_stage(threshold, image_size)
    batch_sensitivity = profile_batch_sensitivity(threshold, image_size)

    print_summary(info, per_stage, batch_sensitivity)

    output = {
        "system": info,
        "threshold": threshold,
        "image_size": image_size,
        "per_stage_latency": per_stage,
        "batch_sensitivity": batch_sensitivity,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
