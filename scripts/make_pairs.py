

import argparse
import csv
import os
import random
import sys

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



SPLIT_SPECS = {
    "train": {"n_identities": 400, "img_min": 5,  "img_max": 30},
    "val":   {"n_identities":  85, "img_min": 5,  "img_max": 20},
    "test":  {"n_identities":  85, "img_min": 5,  "img_max": 20},
}


def _build_identity_pool(split: str, seed: int) -> dict[str, list[str]]:
    """Return a mapping identity_id → [image_path, ...] for the given split.

    Identities and image counts are fully determined by seed so that the same
    config always produces the same population.
    """
    rng = random.Random(seed)
    spec = SPLIT_SPECS[split]
    pool: dict[str, list[str]] = {}
    for idx in range(spec["n_identities"]):
        identity = f"identity_{idx:04d}"
        n_imgs = rng.randint(spec["img_min"], spec["img_max"])
        pool[identity] = [
            f"lfw/{identity}/{identity}_{i:04d}.jpg" for i in range(n_imgs)
        ]
    return pool



def _generate_pairs_v1(
    pool: dict[str, list[str]],
    split: str,
    rng: random.Random,
    pos_per_identity: int,
    neg_per_identity: int,
) -> list[tuple]:
    rows: list[tuple] = []
    identities = list(pool.keys())

    for identity in identities:
        imgs = pool[identity]

        if len(imgs) >= 2:
            sampled = rng.sample(imgs, min(len(imgs), pos_per_identity + 1))
            for i in range(min(pos_per_identity, len(sampled) - 1)):
                rows.append((sampled[i], sampled[i + 1], 1, split))

        # Negative pairs: pair with a random different-identity image
        other_ids = [x for x in identities if x != identity]
        for _ in range(neg_per_identity):
            other = rng.choice(other_ids)
            img_a = rng.choice(imgs)
            img_b = rng.choice(pool[other])
            rows.append((img_a, img_b, 0, split))

    rng.shuffle(rows)
    return rows


def _generate_pairs_v2(
    pool: dict[str, list[str]],
    split: str,
    rng: random.Random,
    pos_per_identity: int,
    neg_per_identity: int,
    max_pairs_per_identity: int,
) -> list[tuple]:
    identities = list(pool.keys())
    pos_rows: list[tuple] = []
    neg_rows: list[tuple] = []

    for identity in identities:
        imgs = pool[identity]
        id_pos: list[tuple] = []
        id_neg: list[tuple] = []

        if len(imgs) >= 2:
            sampled = rng.sample(imgs, min(len(imgs), pos_per_identity + 1))
            for i in range(min(pos_per_identity, len(sampled) - 1)):
                id_pos.append((sampled[i], sampled[i + 1], 1, split))

        other_ids = [x for x in identities if x != identity]
        for _ in range(neg_per_identity):
            other = rng.choice(other_ids)
            img_a = rng.choice(imgs)
            img_b = rng.choice(pool[other])
            if img_a != img_b:
                id_neg.append((img_a, img_b, 0, split))

        per_id_budget = max_pairs_per_identity // 2
        pos_rows.extend(id_pos[:per_id_budget])
        neg_rows.extend(id_neg[:per_id_budget])

    n = min(len(pos_rows), len(neg_rows))
    rng.shuffle(pos_rows)
    rng.shuffle(neg_rows)
    rows = pos_rows[:n] + neg_rows[:n]
    rng.shuffle(rows)
    return rows



def main(config_path: str, version: str) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed: int = config["seed"]

    if version == "v1":
        out_dir = config["data"]["pairs_dir"]
        pos_per = config.get("pair_policy", {}).get("positives_per_identity", 5)
        neg_per = config.get("pair_policy", {}).get("negatives_per_identity", 5)
    elif version == "v2":
        out_dir = config["data"]["pairs_dir_v2"]
        policy = config.get("pair_policy_v2", {})
        pos_per = policy.get("positives_per_identity", 4)
        neg_per = policy.get("negatives_per_identity", 4)
        max_cap = policy.get("max_pairs_per_identity", 8)
    else:
        raise ValueError(f"Unknown version '{version}'. Use 'v1' or 'v2'.")

    os.makedirs(out_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        split_seed = seed + hash(split) % 10000 
        pool = _build_identity_pool(split, split_seed)
        rng = random.Random(split_seed)

        if version == "v1":
            rows = _generate_pairs_v1(pool, split, rng, pos_per, neg_per)
        else:
            rows = _generate_pairs_v2(pool, split, rng, pos_per, neg_per, max_cap)

        out_path = os.path.join(out_dir, f"{split}.csv")
        with open(out_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["left_path", "right_path", "label", "split"])
            writer.writerows(rows)

        n_pos = sum(1 for r in rows if r[2] == 1)
        n_neg = sum(1 for r in rows if r[2] == 0)
        print(f"  [{split}] {len(rows)} pairs  (pos={n_pos}, neg={n_neg})  -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate face-pair CSVs.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--version", choices=["v1", "v2"], default="v1",
        help="Pair generation version: v1=baseline, v2=data-centric improvement.",
    )
    args = parser.parse_args()
    print(f"Generating pairs version={args.version} ...")
    main(args.config, args.version)
    print("Done.")
