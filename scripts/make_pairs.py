import argparse
import os
import random
import sys

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pairs import (
    build_identity_pool,
    generate_pairs_v1,
    generate_pairs_v2,
    load_split_map,
    write_pairs_csv,
)


def main(config_path: str, version: str) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed: int = config["seed"]
    data_dir: str = config["data"]["lfw_images_dir"]
    split_map_path: str = config["data"]["split_map_path"]

    split_map = load_split_map(split_map_path)

    if version == "v1":
        out_dir = config["data"]["pairs_dir"]
        pos_per = config["pair_policy"].get("positives_per_identity", 3)
        neg_per = config["pair_policy"].get("negatives_per_identity", 3)
    elif version == "v2":
        out_dir = config["data"]["pairs_dir_v2"]
        policy = config.get("pair_policy_v2", {})
        pos_per = policy.get("positives_per_identity", 3)
        neg_per = policy.get("negatives_per_identity", 3)
        max_cap = policy.get("max_pairs_per_identity", 6)
    else:
        raise ValueError(f"Unknown version '{version}'. Use 'v1' or 'v2'.")

    os.makedirs(out_dir, exist_ok=True)
    print(f"Generating pairs version={version} ...")

    for split in ["train", "val", "test"]:
        split_seed = seed + abs(hash(split)) % 10000
        rng = random.Random(split_seed)

        identities = split_map[split]
        pool = build_identity_pool(data_dir, identities)

        if version == "v1":
            rows = generate_pairs_v1(pool, split, rng, pos_per, neg_per)
        else:
            rows = generate_pairs_v2(pool, split, rng, pos_per, neg_per, max_cap)

        out_path = os.path.join(out_dir, f"{split}.csv")
        write_pairs_csv(rows, out_path)

        n_pos = sum(1 for r in rows if r[2] == 1)
        n_neg = sum(1 for r in rows if r[2] == 0)
        print(f"  [{split}] {len(rows)} pairs  (pos={n_pos}, neg={n_neg})  -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate face-pair CSVs from real LFW images.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--version", choices=["v1", "v2"], default="v1")
    args = parser.parse_args()
    main(args.config, args.version)
