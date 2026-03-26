
import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    compute_metrics,
    select_threshold,
    threshold_sweep,
)
from src.scoring import generate_scores
from src.tracking import load_runs, log_run
from src.validation import validate_pairs_df, validate_scores



@pytest.fixture
def tiny_pairs() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 20
    identities = [f"id_{i:02d}" for i in range(5)]
    rows = []
    for i in range(n // 2):
        id_ = identities[i % len(identities)]
        rows.append({
            "left_path":  f"lfw/{id_}/{id_}_0001.jpg",
            "right_path": f"lfw/{id_}/{id_}_0002.jpg",
            "label": 1,
            "split": "val",
        })
    other_ids = identities[1:] + identities[:1]  
    for i in range(n // 2):
        id_a = identities[i % len(identities)]
        id_b = other_ids[i % len(other_ids)]
        rows.append({
            "left_path":  f"lfw/{id_a}/{id_a}_0001.jpg",
            "right_path": f"lfw/{id_b}/{id_b}_0001.jpg",
            "label": 0,
            "split": "val",
        })
    return pd.DataFrame(rows)



class TestFullPipelineIntegration:

    def test_pipeline_produces_valid_outputs(self, tiny_pairs, tmp_path):
       
        validate_pairs_df(tiny_pairs, split="val")

        
        scores = generate_scores(
            tiny_pairs, seed=42,
            positive_mean=0.70, positive_std=0.15,
            negative_mean=0.30, negative_std=0.15,
        )
        validate_scores(scores, tiny_pairs)
        assert scores.shape == (len(tiny_pairs),)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

        
        thresholds = np.linspace(0.1, 0.9, 9)
        y_true = tiny_pairs["label"].values
        sweep_results = threshold_sweep(y_true, scores, thresholds)
        assert len(sweep_results) == 9
        assert all("balanced_accuracy" in r for r in sweep_results)

        
        selected = select_threshold(sweep_results, rule="max_balanced_accuracy")
        assert 0.0 <= selected <= 1.0

        
        metrics = compute_metrics(y_true, scores, selected)
        assert metrics["tp"] + metrics["fn"] == (tiny_pairs["label"] == 1).sum()
        assert metrics["tn"] + metrics["fp"] == (tiny_pairs["label"] == 0).sum()
        assert 0.0 <= metrics["balanced_accuracy"] <= 1.0

        
        artifacts_dir = str(tmp_path / "runs")
        summary_file = str(tmp_path / "runs_summary.csv")
        run_id = log_run(
            config_name="test_config",
            split="val",
            data_version="v1",
            threshold=selected,
            metrics=metrics,
            note="integration test run",
            artifacts_dir=artifacts_dir,
            summary_file=summary_file,
        )
        assert run_id == "run_001"

        
        runs = load_runs(artifacts_dir)
        assert len(runs) == 1
        run = runs[0]
        assert run["run_id"] == "run_001"
        assert run["split"] == "val"
        assert run["data_version"] == "v1"
        assert abs(run["threshold"] - selected) < 1e-9
        assert run["metrics"]["balanced_accuracy"] == metrics["balanced_accuracy"]

        
        assert os.path.exists(summary_file)
        import csv
        with open(summary_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["run_id"] == "run_001"

    def test_scores_are_deterministic(self, tiny_pairs):
        
        s1 = generate_scores(tiny_pairs, seed=7)
        s2 = generate_scores(tiny_pairs, seed=7)
        np.testing.assert_array_equal(s1, s2)

    def test_different_seeds_give_different_scores(self, tiny_pairs):
        s1 = generate_scores(tiny_pairs, seed=7)
        s2 = generate_scores(tiny_pairs, seed=99)
        assert not np.array_equal(s1, s2)
