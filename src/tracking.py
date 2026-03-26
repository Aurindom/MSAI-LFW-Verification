
import csv
import json
import os
import subprocess
from datetime import datetime, timezone


def _current_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _next_run_id(artifacts_dir: str) -> str:
    existing = [
        f for f in os.listdir(artifacts_dir)
        if f.startswith("run_") and f.endswith(".json")
    ]
    return f"run_{len(existing) + 1:03d}"


def log_run(
    config_name: str,
    split: str,
    data_version: str,
    threshold: float | None,
    metrics: dict,
    note: str,
    artifacts_dir: str,
    summary_file: str,
    extra: dict | None = None,
) -> str:
    
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)

    run_id = _next_run_id(artifacts_dir)
    timestamp = datetime.now(tz=timezone.utc).isoformat()
    commit = _current_commit()

    run = {
        "run_id": run_id,
        "timestamp": timestamp,
        "commit": commit,
        "config": config_name,
        "split": split,
        "data_version": data_version,
        "threshold": threshold,
        "metrics": metrics,
        "note": note,
    }
    if extra:
        run["extra"] = extra

    
    run_path = os.path.join(artifacts_dir, f"{run_id}.json")
    with open(run_path, "w") as f:
        json.dump(run, f, indent=2)

    
    _write_summary_row(run, summary_file)

    return run_id


def load_runs(artifacts_dir: str) -> list[dict]:
    
    if not os.path.isdir(artifacts_dir):
        return []
    runs = []
    for fname in sorted(os.listdir(artifacts_dir)):
        if fname.startswith("run_") and fname.endswith(".json"):
            with open(os.path.join(artifacts_dir, fname)) as f:
                runs.append(json.load(f))
    return runs


def print_summary(artifacts_dir: str) -> None:
    
    runs = load_runs(artifacts_dir)
    if not runs:
        print("No runs found.")
        return
    header = f"{'run_id':<10} {'split':<6} {'data':<4} {'threshold':>10} {'bal_acc':>9} {'note'}"
    print(header)
    print("-" * len(header))
    for r in runs:
        m = r.get("metrics", {})
        bal = m.get("balanced_accuracy", m.get("mean_balanced_accuracy", "—"))
        thr = r.get("threshold")
        thr_str = f"{thr:.3f}" if thr is not None else "sweep"
        bal_str = f"{bal:.4f}" if isinstance(bal, float) else str(bal)
        print(f"{r['run_id']:<10} {r['split']:<6} {r['data_version']:<4} "
              f"{thr_str:>10} {bal_str:>9}  {r['note']}")


def _write_summary_row(run: dict, summary_file: str) -> None:
    m = run.get("metrics", {})
    row = {
        "run_id": run["run_id"],
        "timestamp": run["timestamp"],
        "commit": run["commit"],
        "config": run["config"],
        "split": run["split"],
        "data_version": run["data_version"],
        "threshold": run.get("threshold", ""),
        "balanced_accuracy": m.get("balanced_accuracy", m.get("mean_balanced_accuracy", "")),
        "f1": m.get("f1", ""),
        "accuracy": m.get("accuracy", ""),
        "tpr": m.get("tpr", ""),
        "fpr": m.get("fpr", ""),
        "note": run["note"],
    }
    fieldnames = list(row.keys())
    write_header = not os.path.exists(summary_file)
    with open(summary_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
