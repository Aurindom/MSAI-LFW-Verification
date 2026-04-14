import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion import ingest_lfw


def main(config_path: str) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    ingest_lfw(
        data_dir=config["data"]["lfw_images_dir"],
        manifest_path=config["data"]["manifest_path"],
        split_map_path=config["data"]["split_map_path"],
        min_faces=config["data"].get("min_faces_per_person", 2),
        train_ratio=config["split_policy"]["train_ratio"],
        val_ratio=config["split_policy"]["val_ratio"],
        seed=config["seed"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest LFW dataset.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
