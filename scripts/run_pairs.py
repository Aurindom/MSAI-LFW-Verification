import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest import load_config
from src.pairs import generate_pairs, split_pairs, save_pairs

cfg = load_config()
labels = np.load(Path(cfg["data"]["out_dir"]) / "labels.npy")

pcfg = cfg["pairs"]
pos, neg = generate_pairs(labels, pcfg["n_positive"], pcfg["n_negative"], pcfg["seed"])
train, train_lbl, test, test_lbl = split_pairs(pos, neg, cfg["data"]["test_size"], pcfg["seed"])
save_pairs(train, train_lbl, test, test_lbl, cfg)
