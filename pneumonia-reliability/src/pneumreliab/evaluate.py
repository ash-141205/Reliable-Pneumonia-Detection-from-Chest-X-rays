"""Evaluation: discrimination metrics with bootstrap CIs, calibration (ECE,
temperature scaling), abstention (selective prediction), clinical operating
point, internal vs external comparison, and plots."""
import argparse, os
import numpy as np
import torch, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             roc_auc_score, roc_curve)
from .data import collect, load_manifest, make_loader
from .models import build
from .utils import get_device, save_json


def softmax(l):
    e = np.exp(l - l.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def _safe(f, *a):
    try:
        return float(f(*a))
    except ValueError:      # only one class present
        return float("nan")


def safe_auc(y, p):
    return _safe(roc_auc_score, y, p)


def basic_metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"auroc": safe_auc(y, p), "auprc": _safe(average_precision_score, y, p),
            "sensitivity": float(tp / max(tp + fn, 1)), "specificity": float(tn / max(tn + fp, 1)),
            "precision": float(tp / max(tp + fp, 1)), "f1": float(f1_score(y, pred, zero_division=0)),
            "accuracy": float((pred == y).mean()), "brier": float(((p - y) ** 2).mean())}


def bootstrap_ci(y, p, thr=0.5, n=1000, seed=0):
    rng = np.random.default_rng(seed); rows = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        rows.append(basic_metrics(y[i], p[i], thr))
    if not rows:
        return {}
    return {k: [float(np.nanpercentile([r[k] for r in rows], 2.5)),
                float(np.nanpercentile([r[k] for r in rows], 97.5))] for k in rows[0]}


def reliability_bins(probs, y, bins=10):
    conf = probs.max(1); correct = (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0.5, 1, bins + 1); edges[0] = 0.0
    c, a, w = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            c.append(conf[m].mean()); a.append(correct[m].mean()); w.append(m.mean())
    return np.array(c), np.array(a), np.array(w)


def ece(probs, y, bins=10):
    c, a, w = reliability_bins(probs, y, bins)
    return float((w * np.abs(a - c)).sum()) if len(c) else float("nan")


def fit_temperature(logits, y):
    t = torch.tensor(logits, dtype=torch.float32); yt = torch.tensor(y)
    f = lambda lt: F.cross_entropy(t / float(np.exp(lt)), yt).item()
    return float(np.exp(minimize_scalar(f, bounds=(-3, 3), method="bounded").x))


def abstention_curve(probs, y, covs=(1.0, 0.9, 0.8, 0.7, 0.5)):
    order = np.argsort(-probs.max(1)); out = {}
    for c in covs:
        keep = order[: max(int(len(y) * c), 1)]
        pred = probs[keep].argmax(1); yk = y[keep]; pos = yk == 1
        out[str(c)] = {"accuracy": float((pred == yk).mean()),
                       "sensitivity": float((pred[pos] == 1).mean()) if pos.any() else None}
    return out


def threshold_at_sensitivity(y, p, target=0.95):
    """Operating point chosen on VALIDATION data only: threshold so that at
    least `target` of pneumonia cases are flagged (screening setting)."""
    pos = p[y == 1]
    return float(np.quantile(pos, 1 - target)) if len(pos) else 0.5


@torch.no_grad()
def get_logits(model, items, device, batch_size=64, num_workers=2):
    model.eval(); L, Y = [], []
    for x, y in make_loader(items, False, False, batch_size, num_workers):
        L.append(model(x.to(device)).cpu()); Y.append(y)
    return torch.cat(L).numpy(), torch.cat(Y).numpy()


def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device)
    m = build(ck["arch"], pretrained=False)
    m.load_state_dict(ck["state_dict"]); m.to(device).eval()
    return m, ck["T"], ck["arch"]


def evaluate_split(logits, y, T, thr_op):
    out = {"temperature": T, "n": int(len(y)), "prevalence": float(y.mean())}
    for tag, t in (("raw", 1.0), ("calibrated", T)):
        pr = softmax(logits / t); p = pr[:, 1]
        m = basic_metrics(y, p)
        m["ece"] = ece(pr, y); m["ci95"] = bootstrap_ci(y, p)
        m["abstention"] = abstention_curve(pr, y)
        out[tag] = m
    pc = softmax(logits / T)[:, 1]
    out["operating_point"] = {"threshold": thr_op, **basic_metrics(y, pc, thr_op)}
    return out


def make_plots(path, store):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for name, (lg, y, T) in store.items():
        if len(set(y)) == 2:
            fpr, tpr, _ = roc_curve(y, softmax(lg / T)[:, 1]); ax[0].plot(fpr, tpr, label=name)
        for tag, t in (("raw", 1.0), ("calibrated", T)):
            c, a, _ = reliability_bins(softmax(lg / t), y)
            ax[1].plot(c, a, marker="o", label=f"{name} ({tag})")
        covs = np.linspace(1, 0.3, 8); pr = softmax(lg / T)
        ax[2].plot(covs, [abstention_curve(pr, y, (c,))[str(c)]["accuracy"] for c in covs],
                   marker="o", label=name)
    ax[0].plot([0, 1], [0, 1], "k--", lw=.8); ax[1].plot([0.5, 1], [0.5, 1], "k--", lw=.8)
    for a_, (t, xl, yl) in zip(ax, [("ROC", "FPR", "TPR"), ("Reliability", "Confidence", "Accuracy"),
                                    ("Abstention", "Coverage", "Accuracy on retained")]):
        a_.set_title(t); a_.set_xlabel(xl); a_.set_ylabel(yl); a_.legend(fontsize=7)
    ax[2].invert_xaxis(); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def evaluate_run(run_dir, external_root=None, device=None, num_workers=2):
    device = device or get_device()
    man = load_manifest(os.path.join(run_dir, "split.csv"))
    model, T, arch = load_checkpoint(os.path.join(run_dir, "model.pt"), device)
    lv, yv = get_logits(model, man["val"], device, num_workers=num_workers)
    thr = threshold_at_sensitivity(yv, softmax(lv / T)[:, 1])
    sets = {"internal_test": man["test"]}
    if external_root:
        sets["external_test"] = collect(external_root)
    store, res = {}, {"arch": arch}
    for name, items in sets.items():
        lg, y = get_logits(model, items, device, num_workers=num_workers)
        store[name] = (lg, y, T); res[name] = evaluate_split(lg, y, T, thr)
        r = res[name]["calibrated"]
        print(f"[{name}] AUROC={r['auroc']:.3f} sens={r['sensitivity']:.3f} "
              f"spec={r['specificity']:.3f} ECE raw={res[name]['raw']['ece']:.3f} -> cal={r['ece']:.3f}")
    save_json(res, os.path.join(run_dir, "metrics.json"))
    make_plots(os.path.join(run_dir, "plots.png"), store)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate a trained run directory")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--external")
    ap.add_argument("--num-workers", type=int, default=2)
    a = ap.parse_args(argv)
    evaluate_run(a.run_dir, a.external, num_workers=a.num_workers)


if __name__ == "__main__":
    main()
