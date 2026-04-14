import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation import compute_roc_curve, compute_auc, select_threshold, threshold_sweep
from src.scoring import score_pairs
from src.tracking import log_run, print_summary
from src.validation import validate_pairs_df, validate_scores, validate_config

import numpy as np
import pandas as pd


def load_pairs(pairs_dir: str, split: str) -> pd.DataFrame:
    path = os.path.join(pairs_dir, f"{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run make_pairs.py first:\n"
            f"  python scripts/make_pairs.py --config configs/m2.yaml --version v1\n"
            f"  python scripts/make_pairs.py --config configs/m2.yaml --version v2"
        )
    df = pd.read_csv(path)
    validate_pairs_df(df, split=split)
    return df[df["split"] == split].reset_index(drop=True)


def run_sweep(df, config, split, data_version, note, artifacts_dir, summary_file, config_name):
    image_size = config["scoring"].get("image_size", 64)
    scores = score_pairs(df, image_size=image_size)
    validate_scores(scores, df)
    y_true = df["label"].values

    sweep_cfg = config["threshold_sweep"]
    thresholds = np.linspace(sweep_cfg["min"], sweep_cfg["max"], sweep_cfg["steps"])
    sweep_results = threshold_sweep(y_true, scores, thresholds)

    rule = config["threshold_selection"]["rule"]
    sel_thr = select_threshold(sweep_results, rule=rule)

    fpr, tpr, _ = compute_roc_curve(y_true, scores)
    auc_val = compute_auc(fpr, tpr)

    from src.evaluation import compute_metrics
    m = compute_metrics(y_true, scores, sel_thr)

    metrics_to_log = {
        "selected_threshold": sel_thr,
        "roc_auc": round(auc_val, 6),
        "balanced_accuracy": m["balanced_accuracy"],
        "tpr": m["tpr"],
        "fpr": m["fpr"],
        "f1": m["f1"],
    }

    run_id = log_run(
        config_name=config_name, split=split, data_version=data_version,
        threshold=sel_thr, metrics=metrics_to_log, note=note,
        artifacts_dir=artifacts_dir, summary_file=summary_file,
        extra={"roc_auc": auc_val},
    )

    _try_save_roc(fpr, tpr, auc_val, data_version, split, config)

    print(f"    [{run_id}] AUC={auc_val:.4f}  selected_threshold={sel_thr:.4f}  "
          f"bal_acc={m['balanced_accuracy']:.4f}")
    return sel_thr, scores, y_true


def run_fixed(df, config, split, data_version, threshold, note, artifacts_dir, summary_file, config_name):
    from src.evaluation import compute_metrics
    image_size = config["scoring"].get("image_size", 64)
    scores = score_pairs(df, image_size=image_size)
    validate_scores(scores, df)
    y_true = df["label"].values
    m = compute_metrics(y_true, scores, threshold)

    run_id = log_run(
        config_name=config_name, split=split, data_version=data_version,
        threshold=threshold, metrics=m, note=note,
        artifacts_dir=artifacts_dir, summary_file=summary_file,
    )

    _try_save_cm(m, threshold, data_version, split, config)

    print(f"    [{run_id}] threshold={threshold:.4f}  "
          f"bal_acc={m['balanced_accuracy']:.4f}  "
          f"f1={m['f1']:.4f}  "
          f"TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")
    return m


def _try_save_roc(fpr, tpr, auc_val, data_version, split, config):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figures_dir = config["reports"]["figures_dir"]
        os.makedirs(figures_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, lw=2, color="#1f77b4", label=f"ROC  AUC={auc_val:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax.set_xlabel("False Positive Rate (FPR)")
        ax.set_ylabel("True Positive Rate (TPR)")
        ax.set_title(f"ROC Curve - {data_version} / {split}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = os.path.join(figures_dir, f"roc_{data_version}_{split}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"      ROC -> {path}")
    except Exception as e:
        print(f"      (ROC plot skipped: {e})")


def _try_save_cm(cm_dict, threshold, data_version, split, config):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figures_dir = config["reports"]["figures_dir"]
        os.makedirs(figures_dir, exist_ok=True)
        matrix = np.array([[cm_dict["tn"], cm_dict["fp"]], [cm_dict["fn"], cm_dict["tp"]]])
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred: Different", "Pred: Same"])
        ax.set_yticklabels(["True: Different", "True: Same"])
        for (r, c), val in np.ndenumerate(matrix):
            ax.text(c, r, str(val), ha="center", va="center", fontsize=14,
                    color="white" if val > matrix.max() / 2 else "black")
        ax.set_title(f"Confusion Matrix  threshold={threshold:.3f}  [{data_version}/{split}]")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        path = os.path.join(figures_dir, f"cm_{data_version}_{split}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"      CM  -> {path}")
    except Exception as e:
        print(f"      (CM plot skipped: {e})")


def main(config_path: str) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    validate_config(config, ["seed", "data", "scoring", "threshold_sweep",
                             "threshold_selection", "tracking"])

    artifacts_dir = config["tracking"]["artifacts_dir"]
    summary_file = config["tracking"]["summary_file"]
    config_name = os.path.splitext(os.path.basename(config_path))[0]

    import scripts.make_pairs as _make_pairs
    for version in ["v1", "v2"]:
        d = config["data"]["pairs_dir"] if version == "v1" else config["data"]["pairs_dir_v2"]
        missing = [s for s in ["val", "test"] if not os.path.exists(os.path.join(d, f"{s}.csv"))]
        if missing:
            print(f"Generating {version} pairs ...")
            _make_pairs.main(config_path, version)

    print("\n=== Run 1: Baseline (v1) - threshold sweep on val ===")
    val_v1 = load_pairs(config["data"]["pairs_dir"], "val")
    thr_v1, _, _ = run_sweep(
        val_v1, config, "val", "v1",
        "Baseline v1 threshold sweep on val - select operating threshold",
        artifacts_dir, summary_file, config_name,
    )

    print("\n=== Run 2: Baseline (v1) - fixed-threshold eval on val ===")
    run_fixed(
        val_v1, config, "val", "v1", thr_v1,
        f"Baseline v1 selected-threshold eval on val (thr={thr_v1:.4f})",
        artifacts_dir, summary_file, config_name,
    )

    print("\n=== Run 3: Baseline (v1) - final reporting on test ===")
    test_v1 = load_pairs(config["data"]["pairs_dir"], "test")
    m3 = run_fixed(
        test_v1, config, "test", "v1", thr_v1,
        f"Baseline v1 final test reporting (locked thr={thr_v1:.4f})",
        artifacts_dir, summary_file, config_name,
    )

    print("\n=== Run 4: Improved (v2) - threshold sweep on val ===")
    val_v2 = load_pairs(config["data"]["pairs_dir_v2"], "val")
    thr_v2, _, _ = run_sweep(
        val_v2, config, "val", "v2",
        "Post data-centric change (v2 pairs) threshold sweep on val",
        artifacts_dir, summary_file, config_name,
    )

    print("\n=== Run 5: Improved (v2) - final reporting on test ===")
    test_v2 = load_pairs(config["data"]["pairs_dir_v2"], "test")
    m5 = run_fixed(
        test_v2, config, "test", "v2", thr_v2,
        f"Post data-centric change (v2) final test reporting (thr={thr_v2:.4f})",
        artifacts_dir, summary_file, config_name,
    )

    print("\n=== Summary ===")
    print_summary(artifacts_dir)

    delta_ba = m5["balanced_accuracy"] - m3["balanced_accuracy"]
    delta_f1 = m5["f1"] - m3["f1"]
    print(f"\nData-centric improvement delta balanced_accuracy = {delta_ba:+.4f}")
    print(f"Data-centric improvement delta F1                = {delta_f1:+.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m2.yaml")
    args = parser.parse_args()
    main(args.config)
