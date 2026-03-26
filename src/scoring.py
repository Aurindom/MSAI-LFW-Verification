import numpy as np
import pandas as pd


def generate_scores(
    pairs_df: pd.DataFrame,
    seed: int,
    positive_mean: float = 0.70,
    positive_std: float = 0.15,
    negative_mean: float = 0.30,
    negative_std: float = 0.15,
) -> np.ndarray:
    labels = pairs_df["label"].values
    row_seeds = np.array(
        [
            abs(hash(f"{seed}:{l}:{r}")) % (2**31)
            for l, r in zip(pairs_df["left_path"], pairs_df["right_path"])
        ],
        dtype=np.int64,
    )

    scores = np.empty(len(pairs_df), dtype=np.float64)
    for i, (label, rseed) in enumerate(zip(labels, row_seeds)):
        rng = np.random.default_rng(rseed)
        if label == 1:
            scores[i] = rng.normal(positive_mean, positive_std)
        else:
            scores[i] = rng.normal(negative_mean, negative_std)

    return np.clip(scores, 0.0, 1.0)
