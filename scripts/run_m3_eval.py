import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.embeddings import cosine_similarity, embed, preprocess
from src.evaluation import compute_auc, compute_metrics, compute_roc_curve, select_threshold, threshold_sweep
from src.tracking import log_run


def score_pairs_embedding(pairs_df: pd.DataFrame, image_size: int = 160) -> np.ndarray:
    scores = []
    n = len(pairs_df)
    for i, row in enumerate(pairs_df.itertuples(index=False), 1):
        if i % 100 == 0 or i == 1:
            print(f"  Scoring pair {i}/{n} ...", flush=True)
        t_a = preprocess(row.left_path, image_size)
        t_b = preprocess(row.right_path, image_size)
        e_a = embed(t_a)
        e_b = embed(t_b)
        raw = cosine_similarity(e_a, e_b)
        scores.append(float(np.clip((raw + 1.0) / 2.0, 0.0, 1.0)))
    return np.array(scores)


def main(config_path: str) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    image_size = config["inference"]["image_size"]
    data_version = config["threshold"]["data_version"]
    pairs_dir = config["data"]["pairs_dir_v2"] if data_version == "v2" else config["data"]["pairs_dir"]
    sweep_split = config["threshold"]["selection_split"]
    rule = config["threshold"]["selection_rule"]
    artifacts_dir = config["tracking"]["artifacts_dir"]
    summary_file = config["tracking"]["summary_file"]

    val_path = os.path.join(pairs_dir, f"{sweep_split}.csv")
    test_path = os.path.join(pairs_dir, "test.csv")

    print(f"\n=== M3 Threshold Selection on {sweep_split} (embedding: FaceNet VGGFace2) ===")
    val_df = pd.read_csv(val_path)
    val_df = val_df[val_df["split"] == sweep_split].reset_index(drop=True)
    print(f"  Pairs: {len(val_df)}  (pos={( val_df['label']==1).sum()}, neg={(val_df['label']==0).sum()})")

    print("  Computing embeddings for val split...")
    val_scores = score_pairs_embedding(val_df, image_size)
    y_val = val_df["label"].values

    thresholds = np.linspace(0.01, 0.99, 99)
    sweep_results = threshold_sweep(y_val, val_scores, thresholds)
    selected_threshold = select_threshold(sweep_results, rule=rule)

    fpr, tpr, _ = compute_roc_curve(y_val, val_scores)
    auc_val = compute_auc(fpr, tpr)
    val_metrics = compute_metrics(y_val, val_scores, selected_threshold)

    print(f"  AUC={auc_val:.4f}  selected_threshold={selected_threshold:.4f}  bal_acc={val_metrics['balanced_accuracy']:.4f}")

    run_id = log_run(
        config_name="m3", split=sweep_split, data_version=data_version,
        threshold=selected_threshold, metrics=val_metrics,
        note=f"M3 FaceNet threshold sweep on {sweep_split} - select operating threshold",
        artifacts_dir=artifacts_dir, summary_file=summary_file,
    )
    print(f"  Logged as {run_id}")

    print(f"\n=== M3 Final Reporting on test (threshold={selected_threshold:.4f}) ===")
    test_df = pd.read_csv(test_path)
    test_df = test_df[test_df["split"] == "test"].reset_index(drop=True)
    print(f"  Pairs: {len(test_df)}  (pos={( test_df['label']==1).sum()}, neg={(test_df['label']==0).sum()})")

    print("  Computing embeddings for test split...")
    test_scores = score_pairs_embedding(test_df, image_size)
    y_test = test_df["label"].values
    test_metrics = compute_metrics(y_test, test_scores, selected_threshold)

    print(f"  threshold={selected_threshold:.4f}  bal_acc={test_metrics['balanced_accuracy']:.4f}  "
          f"f1={test_metrics['f1']:.4f}  "
          f"TP={test_metrics['tp']} FP={test_metrics['fp']} "
          f"TN={test_metrics['tn']} FN={test_metrics['fn']}")

    run_id = log_run(
        config_name="m3", split="test", data_version=data_version,
        threshold=selected_threshold, metrics=test_metrics,
        note=f"M3 FaceNet final test reporting (thr={selected_threshold:.4f})",
        artifacts_dir=artifacts_dir, summary_file=summary_file,
    )
    print(f"  Logged as {run_id}")

    config["threshold"]["value"] = round(float(selected_threshold), 4)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"\nUpdated {config_path} with threshold={selected_threshold:.4f}")

    return selected_threshold, val_metrics, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M3 threshold re-selection and eval with FaceNet embeddings.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
