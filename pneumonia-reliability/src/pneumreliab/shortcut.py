"""Shortcut audit: can a simple model tell which DATASET an image came from
using only border pixels (lung area masked)? High AUROC => non-pathology
source cues exist. Caveat: datasets also differ by age group, so this shows
source cues, not one specific shortcut."""
import argparse, random
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from .data import collect
from .utils import save_json


def _feats(items, n, size, border, rng, mask):
    X = []
    for p, _, _ in rng.sample(items, min(n, len(items))):
        im = np.asarray(Image.open(p).convert("L").resize((size, size)), dtype=np.float32) / 255
        if mask:
            im = im.copy(); im[border:size - border, border:size - border] = 0
        X.append(im.ravel())
    return np.array(X)


def shortcut_audit(a_items, b_items, n=1500, size=64, border=12, seed=0):
    out = {}
    for name, mask in (("border_only", True), ("full_image", False)):
        rng = random.Random(seed)
        A = _feats(a_items, n, size, border, rng, mask); B = _feats(b_items, n, size, border, rng, mask)
        k = min(len(A), len(B)); X = np.vstack([A[:k], B[:k]]); y = np.r_[np.zeros(k), np.ones(k)]
        p = cross_val_predict(LogisticRegression(max_iter=2000, C=0.1), X, y, cv=5,
                              method="predict_proba")[:, 1]
        out[name + "_source_auroc"] = float(roc_auc_score(y, p))
        out["n_per_source"] = int(k)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--internal", required=True); ap.add_argument("--external", required=True)
    ap.add_argument("--out", default="results/shortcut_audit.json")
    a = ap.parse_args(argv)
    res = shortcut_audit(collect(a.internal), collect(a.external))
    print(res); save_json(res, a.out)


if __name__ == "__main__":
    main()
