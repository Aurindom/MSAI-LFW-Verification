import json
import yaml
import numpy as np
from pathlib import Path
from sklearn.datasets import fetch_lfw_people

ROOT = Path(__file__).parent.parent


def load_config(path="configs/config.yaml"):
    with open(ROOT / path) as f:
        return yaml.safe_load(f)


def ingest(cfg):
    dcfg = cfg["data"]
    out_dir = Path(dcfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = fetch_lfw_people(
        data_home=dcfg["data_home"],
        min_faces_per_person=dcfg["min_faces_per_person"],
        resize=0.5,
        color=False,
    )

    images, labels, names = dataset.images, dataset.target, dataset.target_names

    np.save(out_dir / "images.npy", images)
    np.save(out_dir / "labels.npy", labels)
    np.save(out_dir / "names.npy", names)

    manifest = {
        "n_samples": int(len(labels)),
        "n_classes": int(len(names)),
        "image_shape": list(images.shape[1:]),
        "min_faces_per_person": dcfg["min_faces_per_person"],
        "seed": dcfg["seed"],
        "test_size": dcfg["test_size"],
        "split_policy": "random split, fixed seed",
    }

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Ingested {manifest['n_samples']} samples, {manifest['n_classes']} identities → {out_dir}")
    return images, labels, names


if __name__ == "__main__":
    ingest(load_config())
