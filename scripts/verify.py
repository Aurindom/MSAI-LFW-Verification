import argparse
import csv
import os
import sys

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference import verify_pair


def _print_result(result, pair_idx: int = None) -> None:
    header = f"Pair {pair_idx}" if pair_idx is not None else "Pair"
    print(f"\n{header}: {os.path.basename(result.img_a)} | {os.path.basename(result.img_b)}")
    print(f"  Score:      {result.score:.4f}")
    print(f"  Threshold:  {result.threshold:.4f}")
    print(f"  Decision:   {result.decision}")
    print(f"  Confidence: {result.confidence:.4f}")
    print(f"  Latency:    {result.latency_s * 1000:.1f} ms"
          f"  (preprocess={result.preprocess_s*1000:.1f}ms"
          f"  embed={result.embed_s*1000:.1f}ms"
          f"  score={result.score_s*1000:.2f}ms)")


def run_single(img1: str, img2: str, threshold: float, image_size: int) -> None:
    print("Running pair-level inference...")
    result = verify_pair(img1, img2, threshold, image_size)
    _print_result(result)


def run_batch(pairs_csv: str, threshold: float, image_size: int, limit: int) -> None:
    print(f"Running batch inference from {pairs_csv} ...")
    with open(pairs_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if limit:
        rows = rows[:limit]

    n_same = n_diff = n_fail = 0
    for i, row in enumerate(rows, 1):
        try:
            result = verify_pair(row["left_path"], row["right_path"], threshold, image_size)
            _print_result(result, pair_idx=i)
            if result.decision == "SAME":
                n_same += 1
            else:
                n_diff += 1
        except Exception as e:
            print(f"  [FAIL] pair {i}: {e}")
            n_fail += 1

    print(f"\nBatch summary: {len(rows)} pairs — SAME={n_same}  DIFFERENT={n_diff}  FAILED={n_fail}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LFW Face Verifier CLI — Milestone 3\n"
                    "Outputs: score, threshold, decision, confidence, latency per pair.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to m3.yaml config.")
    parser.add_argument("--img1", help="Path to first image (single-pair mode).")
    parser.add_argument("--img2", help="Path to second image (single-pair mode).")
    parser.add_argument("--pairs", help="Path to pairs CSV (batch mode).")
    parser.add_argument("--limit", type=int, default=0, help="Max pairs to run in batch mode (0=all).")
    parser.add_argument("--threshold", type=float, default=None, help="Override threshold from config.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    threshold = args.threshold if args.threshold is not None else config["threshold"]["value"]
    image_size = config["inference"]["image_size"]

    print(f"Embedding model : {config['embedding']['model']} (pretrained={config['embedding']['pretrained']})")
    print(f"Threshold       : {threshold:.4f}  (rule: {config['threshold']['selection_rule']})")
    print(f"Confidence def  : distance from threshold, normalised by room on predicted side — range [0, 1]")

    if args.img1 and args.img2:
        run_single(args.img1, args.img2, threshold, image_size)
    elif args.pairs:
        run_batch(args.pairs, threshold, image_size, args.limit)
    else:
        parser.error("Provide --img1 and --img2 for single-pair mode, or --pairs for batch mode.")


if __name__ == "__main__":
    main()
