import numpy as np
import pandas as pd
import pytest

from src.validation import (
    validate_config,
    validate_metrics_dict,
    validate_no_leakage,
    validate_pairs_df,
    validate_scores,
    validate_threshold,
)


def _make_df(**overrides):
    base = {
        "left_path":  ["a/1.jpg", "a/2.jpg", "b/1.jpg"],
        "right_path": ["a/2.jpg", "b/1.jpg", "c/1.jpg"],
        "label":      [1, 0, 0],
        "split":      ["val", "val", "val"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


class TestValidatePairsDF:
    def test_valid_df_passes(self):
        validate_pairs_df(_make_df()) 

    def test_missing_column_raises(self):
        df = _make_df()
        df = df.drop(columns=["label"])
        with pytest.raises(ValueError, match="missing columns"):
            validate_pairs_df(df)

    def test_invalid_label_raises(self):
        df = _make_df(label=[1, 2, 0])  
        with pytest.raises(ValueError, match="Invalid label values"):
            validate_pairs_df(df)

    def test_invalid_split_raises(self):
        df = _make_df(split=["val", "production", "val"])
        with pytest.raises(ValueError, match="Invalid split values"):
            validate_pairs_df(df)

    def test_self_comparison_raises(self):
        df = _make_df(left_path=["a/1.jpg", "a/1.jpg", "b/1.jpg"],
                      right_path=["a/1.jpg", "b/1.jpg", "c/1.jpg"])
        with pytest.raises(ValueError, match="self-comparisons"):
            validate_pairs_df(df)

    def test_no_positives_raises(self):
        df = _make_df(label=[0, 0, 0])
        with pytest.raises(ValueError, match="No positive pairs"):
            validate_pairs_df(df)

    def test_no_negatives_raises(self):
        df = _make_df(label=[1, 1, 1])
        with pytest.raises(ValueError, match="No negative pairs"):
            validate_pairs_df(df)

    def test_split_filter(self):
        df = _make_df(split=["train", "val", "test"])
        with pytest.raises(ValueError):
            validate_pairs_df(df, split="val")



class TestValidateScores:
    def test_valid_scores_pass(self):
        df = _make_df()
        scores = np.array([0.8, 0.3, 0.2])
        validate_scores(scores, df)  

    def test_wrong_length_raises(self):
        df = _make_df()
        scores = np.array([0.8, 0.3])  
        with pytest.raises(ValueError, match="length"):
            validate_scores(scores, df)

    def test_nan_raises(self):
        df = _make_df()
        scores = np.array([0.8, float("nan"), 0.2])
        with pytest.raises(ValueError, match="non-finite"):
            validate_scores(scores, df)

    def test_inf_raises(self):
        df = _make_df()
        scores = np.array([0.8, float("inf"), 0.2])
        with pytest.raises(ValueError, match="non-finite"):
            validate_scores(scores, df)

    def test_out_of_range_raises(self):
        df = _make_df()
        scores = np.array([1.5, 0.3, 0.2])
        with pytest.raises(ValueError, match="outside expected"):
            validate_scores(scores, df)



class TestValidateThreshold:
    def test_valid_threshold(self):
        validate_threshold(0.5)   
        validate_threshold(0.0)
        validate_threshold(1.0)

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="in \\[0, 1\\]"):
            validate_threshold(1.5)

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="finite"):
            validate_threshold(float("nan"))

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError, match="numeric"):
            validate_threshold("high")



class TestValidateConfig:
    def test_valid_config_passes(self):
        cfg = {"seed": 42, "data": {}}
        validate_config(cfg, required_keys=["seed", "data"])

    def test_missing_key_raises(self):
        cfg = {"seed": 42}
        with pytest.raises(ValueError, match="missing required keys"):
            validate_config(cfg, required_keys=["seed", "data"])


class TestValidateNoLeakage:
    def test_no_overlap_passes(self):
        val_df = pd.DataFrame({"left_path": ["a/1.jpg"], "right_path": ["a/2.jpg"]})
        test_df = pd.DataFrame({"left_path": ["b/1.jpg"], "right_path": ["b/2.jpg"]})
        validate_no_leakage(val_df, test_df)

    def test_overlap_raises(self):
        val_df = pd.DataFrame({"left_path": ["a/1.jpg"], "right_path": ["a/2.jpg"]})
        test_df = pd.DataFrame({"left_path": ["a/1.jpg"], "right_path": ["a/2.jpg"]})
        with pytest.raises(ValueError, match="Split leakage"):
            validate_no_leakage(val_df, test_df)



class TestValidateMetricsDict:
    def test_valid_metrics_pass(self):
        m = {"accuracy": 0.9, "tp": 10, "fp": 2, "tn": 8, "fn": 1}
        validate_metrics_dict(m)

    def test_none_value_raises(self):
        m = {"accuracy": None}
        with pytest.raises(ValueError, match="None values"):
            validate_metrics_dict(m)

    def test_negative_count_raises(self):
        m = {"tp": -1, "fp": 2, "tn": 8, "fn": 1}
        with pytest.raises(ValueError, match="negative"):
            validate_metrics_dict(m)
