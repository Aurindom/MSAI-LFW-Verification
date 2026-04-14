import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation import (
    compute_metrics,
    compute_roc_curve,
    compute_auc,
    select_threshold,
    threshold_sweep,
)
from src.scoring import score_pairs
from src.tracking import log_run, print_summary
from src.validation import (
    validate_config,
    validate_pairs_df,
    validate_scores,
    validate_threshold,
    validate_metrics_dict,
)


def _save_roc_plot(fpr, tpr, auc_val: float, out_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, color="#1f77b4", label=f"ROC  AUC={auc_val:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("ROC Curve – Face Verification")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ROC plot saved → {out_path}")


def _save_confusion_matrix_plot(cm: dict, threshold: float, out_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: Different", "Pred: Same"])
    ax.set_yticklabels(["True: Different", "True: Same"])
    for (r, c), val in np.ndenumerate(matrix):
        ax.text(c, r, str(val), ha="center", va="center", fontsize=14,
                color="white" if val > matrix.max() / 2 else "black")
    ax.set_title(f"Confusion Matrix  (threshold={threshold:.3f})")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved → {out_path}")


def main(args) -> None:
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    validate_config(config, ["seed", "data", "scoring", "threshold_sweep",
                             "threshold_selection", "tracking", "reports"])

    seed: int = config["seed"]
    split: str = args.split
    data_version: str = args.data_version

    
    if data_version == "v1":
        pairs_dir = config["data"]["pairs_dir"]
    elif data_version == "v2":
        pairs_dir = config["data"]["pairs_dir_v2"]
    else:
        raise ValueError(f"Unknown data version '{data_version}'.")

    pairs_path = os.path.join(pairs_dir, f"{split}.csv")
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(
            f"Pairs file not found: {pairs_path}\n"
            f"Run: python scripts/make_pairs.py --config {args.config} "
            f"--version {data_version}"
        )

    
    df = pd.read_csv(pairs_path)
    validate_pairs_df(df, split=split)
    split_df = df[df["split"] == split].reset_index(drop=True)
    print(f"  Loaded {len(split_df)} pairs for split='{split}' "
          f"(pos={( split_df['label']==1).sum()}, "
          f"neg={(split_df['label']==0).sum()})")

    
    image_size = config["scoring"].get("image_size", 64)
    scores = score_pairs(split_df, image_size=image_size)
    validate_scores(scores, split_df)
    y_true = split_df["label"].values

  
    sweep_cfg = config["threshold_sweep"]
    thresholds = np.linspace(sweep_cfg["min"], sweep_cfg["max"], sweep_cfg["steps"])
    sweep_results = threshold_sweep(y_true, scores, thresholds)

    figures_dir = config["reports"]["figures_dir"]
    artifacts_dir = config["tracking"]["artifacts_dir"]
    summary_file = config["tracking"]["summary_file"]
    config_name = os.path.splitext(os.path.basename(args.config))[0]

    if args.threshold is None:
        
        rule = config["threshold_selection"]["rule"]
        selected_threshold = select_threshold(sweep_results, rule=rule)
        print(f"  Threshold selection rule: {rule}")
        print(f"  Selected threshold: {selected_threshold:.4f}")

       
        fpr, tpr, _ = compute_roc_curve(y_true, scores)
        auc_val = compute_auc(fpr, tpr)
        print(f"  ROC AUC: {auc_val:.4f}")

        selected_metrics = compute_metrics(y_true, scores, selected_threshold)
        validate_metrics_dict(selected_metrics)

        
        sweep_bal_accs = [r["balanced_accuracy"] for r in sweep_results]
        metrics_to_log = {
            "selected_threshold": selected_threshold,
            "roc_auc": round(auc_val, 6),
            "mean_balanced_accuracy": round(float(np.mean(sweep_bal_accs)), 6),
            "max_balanced_accuracy": round(float(np.max(sweep_bal_accs)), 6),
            "balanced_accuracy": selected_metrics["balanced_accuracy"],
            "tpr": selected_metrics["tpr"],
            "fpr": selected_metrics["fpr"],
            "f1": selected_metrics["f1"],
        }
        extra = {"sweep_n_thresholds": len(sweep_results), "auc": auc_val}

        run_id = log_run(
            config_name=config_name, split=split, data_version=data_version,
            threshold=selected_threshold, metrics=metrics_to_log,
            note=args.note, artifacts_dir=artifacts_dir, summary_file=summary_file,
            extra=extra,
        )

        suffix = f"{data_version}_{split}"
        _save_roc_plot(fpr, tpr, auc_val,
                       os.path.join(figures_dir, f"roc_{suffix}.png"))
        _save_confusion_matrix_plot(
            selected_metrics, selected_threshold,
            os.path.join(figures_dir, f"cm_{suffix}.png"),
        )

    else:        
        threshold = float(args.threshold)
        validate_threshold(threshold)
        metrics = compute_metrics(y_true, scores, threshold)
        validate_metrics_dict(metrics)

        print(f"  Threshold (fixed): {threshold:.4f}")
        for k in ("balanced_accuracy", "f1", "tpr", "fpr", "accuracy"):
            print(f"    {k}: {metrics[k]:.4f}")
        print(f"    Confusion: TP={metrics['tp']} FP={metrics['fp']} "
              f"TN={metrics['tn']} FN={metrics['fn']}")

        run_id = log_run(
            config_name=config_name, split=split, data_version=data_version,
            threshold=threshold, metrics=metrics,
            note=args.note, artifacts_dir=artifacts_dir, summary_file=summary_file,
        )

        suffix = f"{data_version}_{split}_t{threshold:.3f}"
        _save_confusion_matrix_plot(
            metrics, threshold,
            os.path.join(figures_dir, f"cm_{suffix}.png"),
        )

    print(f"  Logged as {run_id}")

    if args.show_summary:
        print()
        print_summary(artifacts_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a tracked face-verification evaluation.")
    parser.add_argument("--config", default="configs/m2.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--data-version", choices=["v1", "v2"], default="v1",
                        dest="data_version")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Fixed threshold; omit to run a sweep and auto-select.")
    parser.add_argument("--note", default="", help="Short description of this run.")
    parser.add_argument("--show-summary", action="store_true",
                        help="Print run summary table after logging.")
    args = parser.parse_args()
    main(args)
