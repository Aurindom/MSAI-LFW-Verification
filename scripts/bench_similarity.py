import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import argparse
import yaml
import numpy as np
import time
from src.similarity import (
    cosine_similarity_vectorized,
    euclidean_distance_vectorized,
    cosine_similarity_loop,
    euclidean_distance_loop,
)


def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config["seed"]
    N = config["benchmark"]["N"]
    D = config["benchmark"]["D"]

    np.random.seed(seed)

    a = np.random.randn(N, D)
    b = np.random.randn(N, D)

    print("\n=== COSINE SIMILARITY ===")

    start = time.perf_counter()
    cos_loop = cosine_similarity_loop(a, b)
    loop_time = time.perf_counter() - start

    start = time.perf_counter()
    cos_vec = cosine_similarity_vectorized(a, b)
    vec_time = time.perf_counter() - start

    max_diff = np.max(np.abs(cos_loop - cos_vec))

    print(f"Loop time: {loop_time:.4f} sec")
    print(f"Vectorized time: {vec_time:.4f} sec")
    print(f"Speedup: {loop_time/vec_time:.2f}x")
    print(f"Max difference: {max_diff:.10f}")

    print("\n=== EUCLIDEAN DISTANCE ===")

    start = time.perf_counter()
    euc_loop = euclidean_distance_loop(a, b)
    loop_time = time.perf_counter() - start

    start = time.perf_counter()
    euc_vec = euclidean_distance_vectorized(a, b)
    vec_time = time.perf_counter() - start

    max_diff = np.max(np.abs(euc_loop - euc_vec))

    print(f"Loop time: {loop_time:.4f} sec")
    print(f"Vectorized time: {vec_time:.4f} sec")
    print(f"Speedup: {loop_time/vec_time:.2f}x")
    print(f"Max difference: {max_diff:.10f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    main(args.config)