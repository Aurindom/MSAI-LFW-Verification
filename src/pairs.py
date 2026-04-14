import csv
import json
import os
import random


def load_split_map(split_map_path: str) -> dict[str, list[str]]:
    with open(split_map_path) as f:
        return json.load(f)


def build_identity_pool(
    data_dir: str,
    identities: list[str],
) -> dict[str, list[str]]:
    pool: dict[str, list[str]] = {}
    for identity in identities:
        safe = identity.replace(" ", "_")
        id_dir = os.path.join(data_dir, safe)
        if not os.path.isdir(id_dir):
            continue
        imgs = sorted(
            os.path.join(id_dir, f)
            for f in os.listdir(id_dir)
            if f.lower().endswith(".jpg")
        )
        if len(imgs) >= 2:
            pool[identity] = imgs
    return pool


def generate_pairs_v1(
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

        other_ids = [x for x in identities if x != identity]
        for _ in range(neg_per_identity):
            other = rng.choice(other_ids)
            img_a = rng.choice(imgs)
            img_b = rng.choice(pool[other])
            rows.append((img_a, img_b, 0, split))

    rng.shuffle(rows)
    return rows


def generate_pairs_v2(
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

        budget = max_pairs_per_identity // 2
        pos_rows.extend(id_pos[:budget])
        neg_rows.extend(id_neg[:budget])

    n = min(len(pos_rows), len(neg_rows))
    rng.shuffle(pos_rows)
    rng.shuffle(neg_rows)
    rows = pos_rows[:n] + neg_rows[:n]
    rng.shuffle(rows)
    return rows


def write_pairs_csv(rows: list[tuple], out_path: str) -> None:
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["left_path", "right_path", "label", "split"])
        normalized = [
            (str(r[0]).replace("\\", "/"), str(r[1]).replace("\\", "/"), r[2], r[3])
            for r in rows
        ]
        writer.writerows(normalized)
