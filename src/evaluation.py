import numpy as np

def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def compute_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    cm = confusion_counts(y_true, y_pred)
    tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]

    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    tpr = recall
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_accuracy = (tpr + tnr) / 2.0

    return {
        "threshold": float(threshold),
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "balanced_accuracy": round(balanced_accuracy, 6),
        "tpr": round(tpr, 6),
        "fpr": round(fpr, 6),
        "fnr": round(fnr, 6),
        "tnr": round(tnr, 6),
        **cm,
    }



def threshold_sweep(y_true: np.ndarray, scores: np.ndarray,
                    thresholds: np.ndarray) -> list[dict]:
    
    thresholds = np.sort(thresholds)
    return [compute_metrics(y_true, scores, t) for t in thresholds]


def select_threshold(sweep_results: list[dict], rule: str = "max_balanced_accuracy") -> float:
    
    if rule == "max_balanced_accuracy":
        key = "balanced_accuracy"
        best = max(sweep_results, key=lambda r: r[key])
    elif rule == "max_f1":
        key = "f1"
        best = max(sweep_results, key=lambda r: r[key])
    elif rule == "min_fnr":
        key = "fnr"
        best = min(sweep_results, key=lambda r: r[key])
    else:
        raise ValueError(f"Unknown threshold selection rule: '{rule}'")

    return best["threshold"]



def compute_roc_curve(y_true: np.ndarray,
                      scores: np.ndarray,
                      n_points: int = 201) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    thresholds = np.linspace(0.0, 1.0, n_points)
    fpr_arr = np.empty(n_points)
    tpr_arr = np.empty(n_points)
    for i, t in enumerate(thresholds):
        m = compute_metrics(y_true, scores, t)
        fpr_arr[i] = m["fpr"]
        tpr_arr[i] = m["tpr"]
    return fpr_arr, tpr_arr, thresholds


def compute_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    
    order = np.lexsort((tpr, fpr))
    fpr_s, tpr_s = fpr[order], tpr[order]
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    return float(trapz(tpr_s, fpr_s))
