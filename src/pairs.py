import json
import numpy as np
from pathlib import Path


def generate_pairs(labels, n_positive, n_negative, seed):
    rng = np.random.default_rng(seed)
    unique = np.unique(labels)

    pos = []
    for uid in unique:
        idxs = np.where(labels == uid)[0]
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                pos.append((int(idxs[i]), int(idxs[j])))

    pos = np.array(pos)
    chosen = rng.choice(len(pos), size=min(n_positive, len(pos)), replace=False)
    pos = pos[chosen]

    neg = []
    while len(neg) < n_negative:
        a_id, b_id = rng.choice(unique, size=2, replace=False)
        a = int(rng.choice(np.where(labels == a_id)[0]))
        b = int(rng.choice(np.where(labels == b_id)[0]))
        neg.append((a, b))
    neg = np.array(neg)

    return pos, neg


def split_pairs(pos, neg, test_size, seed):
    rng = np.random.default_rng(seed)

    def _split(arr):
        n_test = max(1, int(len(arr) * test_size))
        idx = rng.permutation(len(arr))
        return arr[idx[n_test:]], arr[idx[:n_test]]

    pos_tr, pos_te = _split(pos)
    neg_tr, neg_te = _split(neg)

    train = np.vstack([pos_tr, neg_tr])
    train_labels = np.concatenate([np.ones(len(pos_tr)), np.zeros(len(neg_tr))]).astype(int)
    test = np.vstack([pos_te, neg_te])
    test_labels = np.concatenate([np.ones(len(pos_te)), np.zeros(len(neg_te))]).astype(int)

    return train, train_labels, test, test_labels


def save_pairs(train, train_labels, test, test_labels, cfg):
    out_dir = Path(cfg["pairs"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "train_pairs.npy", train)
    np.save(out_dir / "train_labels.npy", train_labels)
    np.save(out_dir / "test_pairs.npy", test)
    np.save(out_dir / "test_labels.npy", test_labels)

    meta = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_positive_train": int(train_labels.sum()),
        "n_positive_test": int(test_labels.sum()),
        "seed": cfg["pairs"]["seed"],
        "test_size": cfg["data"]["test_size"],
    }
    with open(out_dir / "pairs_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Train: {meta['n_train']} pairs, Test: {meta['n_test']} pairs → {out_dir}")
