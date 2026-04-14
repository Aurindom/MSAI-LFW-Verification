import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference import verify_pair


def _run_one(args):
    left, right, threshold, image_size = args
    t0 = time.perf_counter()
    try:
        result = verify_pair(left, right, threshold, image_size)
        return result.latency_s, None
    except Exception as e:
        return time.perf_counter() - t0, str(e)


def main(config_path: str, n_workers: int = None, n_requests: int = None) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    lt_cfg = config["load_test"]
    pairs_source = lt_cfg["pairs_source"]
    output_path = lt_cfg["output_path"]
    n_workers = n_workers or lt_cfg["n_workers"]
    n_requests = n_requests or lt_cfg["n_requests"]
    threshold = config["threshold"]["value"]
    image_size = config["inference"]["image_size"]

    with open(pairs_source, newline="") as f:
        all_pairs = list(csv.DictReader(f))

    if not all_pairs:
        raise ValueError(f"No pairs found in {pairs_source}")

    pairs = []
    for i in range(n_requests):
        row = all_pairs[i % len(all_pairs)]
        pairs.append((row["left_path"], row["right_path"], threshold, image_size))

    print(f"\n=== Load Test ===")
    print(f"  Workers  : {n_workers}")
    print(f"  Requests : {n_requests}")
    print(f"  Source   : {pairs_source}")
    print(f"  Threshold: {threshold}")
    print("  Running ...")

    latencies = []
    failures = []

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_run_one, p): i for i, p in enumerate(pairs)}
        for fut in as_completed(futures):
            lat, err = fut.result()
            if err:
                failures.append(err)
            else:
                latencies.append(lat)
    wall_time = time.perf_counter() - wall_start

    latencies_arr = np.array(latencies) if latencies else np.array([0.0])
    results = {
        "n_workers": n_workers,
        "n_requests": n_requests,
        "n_success": len(latencies),
        "n_failure": len(failures),
        "wall_time_s": round(wall_time, 3),
        "throughput_rps": round(len(latencies) / wall_time, 3) if wall_time > 0 else 0,
        "latency_mean_ms": round(float(latencies_arr.mean()) * 1000, 2),
        "latency_p50_ms": round(float(np.percentile(latencies_arr, 50)) * 1000, 2),
        "latency_p95_ms": round(float(np.percentile(latencies_arr, 95)) * 1000, 2),
        "latency_p99_ms": round(float(np.percentile(latencies_arr, 99)) * 1000, 2),
        "latency_min_ms": round(float(latencies_arr.min()) * 1000, 2),
        "latency_max_ms": round(float(latencies_arr.max()) * 1000, 2),
        "pairs_source": pairs_source,
    }

    print(f"\n  Results:")
    print(f"    Total requests : {n_requests}  (success={results['n_success']}, failure={results['n_failure']})")
    print(f"    Wall-clock time: {results['wall_time_s']:.2f} s")
    print(f"    Throughput     : {results['throughput_rps']:.2f} req/s")
    print(f"    Latency mean   : {results['latency_mean_ms']:.1f} ms")
    print(f"    Latency p50    : {results['latency_p50_ms']:.1f} ms")
    print(f"    Latency p95    : {results['latency_p95_ms']:.1f} ms")
    print(f"    Latency p99    : {results['latency_p99_ms']:.1f} ms")
    print(f"    Latency min/max: {results['latency_min_ms']:.1f} / {results['latency_max_ms']:.1f} ms")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved -> {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load test: concurrent face verification inference.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", type=int, default=None, help="Override n_workers from config.")
    parser.add_argument("--requests", type=int, default=None, help="Override n_requests from config.")
    args = parser.parse_args()
    main(args.config, n_workers=args.workers, n_requests=args.requests)
