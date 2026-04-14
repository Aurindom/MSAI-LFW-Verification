import csv
import json
import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.evaluation import compute_metrics, select_threshold, threshold_sweep
from src.pairs import write_pairs_csv
from src.scoring import compute_pixel_similarity, score_pairs
from src.tracking import load_runs, log_run
from src.validation import validate_pairs_df, validate_scores


def _make_image(path: str, color: tuple[int, int, int]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.fromarray(
        np.full((62, 47, 3), color, dtype=np.uint8), mode="RGB"
    )
    img.save(path)


@pytest.fixture
def image_pairs(tmp_path) -> pd.DataFrame:
    id_a_img0 = str(tmp_path / "id_a" / "id_a_0000.jpg")
    id_a_img1 = str(tmp_path / "id_a" / "id_a_0001.jpg")
    id_b_img0 = str(tmp_path / "id_b" / "id_b_0000.jpg")
    id_b_img1 = str(tmp_path / "id_b" / "id_b_0001.jpg")

    _make_image(id_a_img0, (200, 100, 50))
    _make_image(id_a_img1, (195, 105, 48))
    _make_image(id_b_img0, (30, 200, 180))
    _make_image(id_b_img1, (28, 198, 175))

    rows = [
        {"left_path": id_a_img0, "right_path": id_a_img1, "label": 1, "split": "val"},
        {"left_path": id_b_img0, "right_path": id_b_img1, "label": 1, "split": "val"},
        {"left_path": id_a_img0, "right_path": id_b_img0, "label": 0, "split": "val"},
        {"left_path": id_a_img1, "right_path": id_b_img1, "label": 0, "split": "val"},
    ]
    return pd.DataFrame(rows)


class TestFullPipelineIntegration:

    def test_pipeline_produces_valid_outputs(self, image_pairs, tmp_path):
        validate_pairs_df(image_pairs, split="val")

        scores = score_pairs(image_pairs, image_size=32)
        validate_scores(scores, image_pairs)
        assert scores.shape == (len(image_pairs),)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

        thresholds = np.linspace(0.1, 0.9, 9)
        y_true = image_pairs["label"].values
        sweep_results = threshold_sweep(y_true, scores, thresholds)
        assert len(sweep_results) == 9
        assert all("balanced_accuracy" in r for r in sweep_results)

        selected = select_threshold(sweep_results, rule="max_balanced_accuracy")
        assert 0.0 <= selected <= 1.0

        metrics = compute_metrics(y_true, scores, selected)
        assert metrics["tp"] + metrics["fn"] == (image_pairs["label"] == 1).sum()
        assert metrics["tn"] + metrics["fp"] == (image_pairs["label"] == 0).sum()
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
        with open(summary_file) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["run_id"] == "run_001"

    def test_scores_are_deterministic(self, image_pairs):
        s1 = score_pairs(image_pairs, image_size=32)
        s2 = score_pairs(image_pairs, image_size=32)
        np.testing.assert_array_equal(s1, s2)

    def test_same_image_gives_max_score(self, image_pairs):
        row = image_pairs.iloc[0]
        score = compute_pixel_similarity(row.left_path, row.left_path, image_size=32)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_scores_in_range(self, image_pairs):
        scores = score_pairs(image_pairs, image_size=32)
        assert (scores >= 0.0).all()
        assert (scores <= 1.0).all()


class TestPairsCsvPortability:

    def test_csv_paths_use_forward_slashes(self, tmp_path):
        left = str(tmp_path / "id_a" / "id_a_0000.jpg")
        right = str(tmp_path / "id_a" / "id_a_0001.jpg")
        out = str(tmp_path / "pairs.csv")

        write_pairs_csv([(left, right, 1, "val")], out)

        with open(out, newline="") as f:
            rows = list(csv.DictReader(f))

        assert "\\" not in rows[0]["left_path"]
        assert "\\" not in rows[0]["right_path"]

    def test_csv_paths_are_valid_posix(self, tmp_path):
        left = str(tmp_path / "id_a" / "img_0.jpg")
        right = str(tmp_path / "id_b" / "img_0.jpg")
        out = str(tmp_path / "pairs.csv")

        write_pairs_csv([(left, right, 0, "test")], out)

        with open(out, newline="") as f:
            rows = list(csv.DictReader(f))

        assert rows[0]["left_path"].count("/") >= 1
        assert rows[0]["right_path"].count("/") >= 1
