import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import yaml
import csv
import os
import random


def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config["seed"]
    random.seed(seed)

    os.makedirs("outputs/pairs", exist_ok=True)

    splits = ["train", "val", "test"]

    for split in splits:
        filepath = f"outputs/pairs/{split}.csv"
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["left_path", "right_path", "label", "split"])

            for i in range(10):  # deterministic small example
                writer.writerow(
                    [f"personA/img{i}.jpg", f"personA/img{i+1}.jpg", 1, split]
                )
                writer.writerow(
                    [f"personA/img{i}.jpg", f"personB/img{i}.jpg", 0, split]
                )

        print(f"Wrote {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    main(args.config)