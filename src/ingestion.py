import json
import os

import numpy as np
from PIL import Image
from sklearn.datasets import fetch_lfw_people


def ingest_lfw(
    data_dir: str = "data/lfw_images",
    manifest_path: str = "outputs/manifest.json",
    split_map_path: str = "outputs/split_map.json",
    min_faces: int = 2,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    print("Fetching LFW via sklearn (cached after first download)...")
    lfw = fetch_lfw_people(min_faces_per_person=min_faces, color=True)

    images = lfw.images
    targets = lfw.target
    target_names = lfw.target_names

    identity_to_images: dict[str, list[np.ndarray]] = {n: [] for n in target_names}
    for img, t in zip(images, targets):
        identity_to_images[target_names[t]].append(img)

    all_identities = sorted(identity_to_images.keys())
    idx = np.arange(len(all_identities))
    np.random.default_rng(seed).shuffle(idx)
    shuffled = [all_identities[i] for i in idx]

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    split_map: dict[str, list[str]] = {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train : n_train + n_val],
        "test":  shuffled[n_train + n_val :],
    }

    counts: dict[str, dict] = {}
    for split_name, identities in split_map.items():
        n_imgs = 0
        for identity in identities:
            safe = identity.replace(" ", "_")
            id_dir = os.path.join(data_dir, safe)
            os.makedirs(id_dir, exist_ok=True)
            for i, arr in enumerate(identity_to_images[identity]):
                img_path = os.path.join(id_dir, f"{safe}_{i:04d}.jpg")
                if not os.path.exists(img_path):
                    uint8 = (arr * 255).astype(np.uint8)
                    Image.fromarray(uint8, mode="RGB").save(img_path)
                n_imgs += 1
        counts[split_name] = {"identities": len(identities), "images": n_imgs}

    with open(split_map_path, "w") as f:
        json.dump(split_map, f, indent=2)

    h, w = images[0].shape[:2]
    manifest = {
        "seed": seed,
        "split_policy": f"identity-based {int(train_ratio*100)}/{int(val_ratio*100)}/{int((1-train_ratio-val_ratio)*100)}",
        "data_source": "sklearn.datasets.fetch_lfw_people (min_faces_per_person=2, color=True)",
        "image_shape": [h, w, 3],
        "total_identities": len(all_identities),
        "total_images": int(len(images)),
        "counts": counts,
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    for s, c in counts.items():
        print(f"  [{s}] {c['identities']} identities, {c['images']} images")
    print(f"Manifest  -> {manifest_path}")
    print(f"Split map -> {split_map_path}")

    return manifest
