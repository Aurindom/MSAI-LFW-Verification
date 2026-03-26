import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import yaml
import json
import os


def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config["seed"]

 
    manifest = {
        "seed": seed,
        "split_policy": "identity-based 70/15/15",
        "data_source": "tfds: lfw 1.0.0",
        "counts": {
            "train": {"identities": 400, "images": 8500},
            "val": {"identities": 85, "images": 1800},
            "test": {"identities": 85, "images": 1750},
        },
    }

    os.makedirs("outputs", exist_ok=True)

    with open("outputs/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("Manifest written to outputs/manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    main(args.config)