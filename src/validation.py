import numpy as np
import pandas as pd


REQUIRED_PAIR_COLUMNS = {"left_path", "right_path", "label", "split"}
VALID_SPLITS = {"train", "val", "test"}
VALID_LABELS = {0, 1}


def validate_pairs_df(df: pd.DataFrame, split: str | None = None) -> None:
    
    missing = REQUIRED_PAIR_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Pairs DataFrame missing columns: {sorted(missing)}")

    bad_labels = set(df["label"].unique()) - VALID_LABELS
    if bad_labels:
        raise ValueError(f"Invalid label values found: {bad_labels}. Expected {{0, 1}}.")

    bad_splits = set(df["split"].unique()) - VALID_SPLITS
    if bad_splits:
        raise ValueError(f"Invalid split values found: {bad_splits}. Expected {VALID_SPLITS}.")

    if split is not None:
        subset = df[df["split"] == split]
        if len(subset) == 0:
            raise ValueError(f"No rows found for split='{split}'.")

    duplicates = df[df["left_path"] == df["right_path"]]
    if len(duplicates) > 0:
        raise ValueError(
            f"Found {len(duplicates)} pairs where left_path == right_path "
            f"(self-comparisons are invalid)."
        )

    if split is not None:
        subset = df[df["split"] == split]
        n_pos = (subset["label"] == 1).sum()
        n_neg = (subset["label"] == 0).sum()
    else:
        n_pos = (df["label"] == 1).sum()
        n_neg = (df["label"] == 0).sum()

    if n_pos == 0:
        raise ValueError("No positive pairs (label=1) found.")
    if n_neg == 0:
        raise ValueError("No negative pairs (label=0) found.")


def validate_scores(scores: np.ndarray, pairs_df: pd.DataFrame) -> None:
    
    if len(scores) != len(pairs_df):
        raise ValueError(
            f"Score array length {len(scores)} != pairs count {len(pairs_df)}."
        )
    if not np.isfinite(scores).all():
        n_bad = (~np.isfinite(scores)).sum()
        raise ValueError(f"Score array contains {n_bad} non-finite value(s) (NaN/Inf).")
    if scores.min() < -0.01 or scores.max() > 1.01:
        raise ValueError(
            f"Score values outside expected [0, 1] range: "
            f"min={scores.min():.4f}, max={scores.max():.4f}."
        )


def validate_threshold(threshold: float) -> None:
    
    if not isinstance(threshold, (int, float)):
        raise ValueError(f"Threshold must be a numeric value, got {type(threshold)}.")
    if not np.isfinite(threshold):
        raise ValueError(f"Threshold must be finite, got {threshold}.")
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"Threshold must be in [0, 1], got {threshold:.4f}.")


def validate_config(config: dict, required_keys: list[str]) -> None:
    
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}.")


def validate_no_leakage(val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    
    val_pairs = set(zip(val_df["left_path"], val_df["right_path"]))
    test_pairs = set(zip(test_df["left_path"], test_df["right_path"]))
    overlap = val_pairs & test_pairs
    if overlap:
        raise ValueError(
            f"Split leakage detected: {len(overlap)} pair(s) appear in both "
            f"val and test sets."
        )


def validate_metrics_dict(metrics: dict) -> None:
    none_keys = [k for k, v in metrics.items() if v is None]
    if none_keys:
        raise ValueError(f"Metrics dict has None values for: {none_keys}.")
    for key in ("tp", "fp", "tn", "fn"):
        if key in metrics and metrics[key] < 0:
            raise ValueError(f"Confusion count '{key}' is negative: {metrics[key]}.")
