import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.similarity import cosine_loop, euclidean_loop, cosine_vectorized, euclidean_vectorized

N, DIM = 500, 2914  # pairs × LFW flattened image size (62×47)
rng = np.random.default_rng(42)
A = rng.random((N, DIM)).astype(np.float32)
B = rng.random((N, DIM)).astype(np.float32)


def timed(fn, *args):
    t = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - t


cos_l,  t_cos_l  = timed(cosine_loop,         A, B)
cos_v,  t_cos_v  = timed(cosine_vectorized,   A, B)
euc_l,  t_euc_l  = timed(euclidean_loop,      A, B)
euc_v,  t_euc_v  = timed(euclidean_vectorized, A, B)

print(f"{'Method':<28} {'Time (s)':>10} {'Speedup':>10}")
print("-" * 50)
print(f"{'cosine    (loop)':<28} {t_cos_l:>10.4f} {'baseline':>10}")
print(f"{'cosine    (vectorized)':<28} {t_cos_v:>10.4f} {t_cos_l / t_cos_v:>9.1f}x")
print(f"{'euclidean (loop)':<28} {t_euc_l:>10.4f} {'baseline':>10}")
print(f"{'euclidean (vectorized)':<28} {t_euc_v:>10.4f} {t_euc_l / t_euc_v:>9.1f}x")

np.testing.assert_allclose(cos_l, cos_v, rtol=1e-5, atol=1e-6)
np.testing.assert_allclose(euc_l, euc_v, rtol=1e-5, atol=1e-6)
print("\nAll correctness checks passed.")
