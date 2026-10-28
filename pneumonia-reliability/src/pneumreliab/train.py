"""Train (and then evaluate) models. Each run gets its own folder:
  <out>/<tag>_<arch>_s<seed>/{model.pt, split.csv, config.json, train_log.json,
                              metrics.json, plots.png}

GAN augmentation flags (optional, fully ablatable)
---------------------------------------------------
--synthetic-dir   Directory produced by `python -m pneumreliab.gan sample`.
                  Images are added to the TRAINING set only.
--synthetic-ratio Fraction of the minority-class real count to add (e.g. 0.5
                  or 1.0).  Ignored if --synthetic-dir is not given.

The validation split and temperature fitting remain real-only.
"""
import argparse, os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from .data import (CLASSES, collect, make_loader, patient_split, save_manifest,
                   collect_synthetic, append_synthetic_to_train)
from .evaluate import evaluate_run, fit_temperature, get_logits, safe_auc, softmax
from .models import ARCHS, build
from .utils import env_info, get_device, save_json, set_seed


def train_model(model, tr, va, cfg, device):
    model.to(device)
    ys = np.array([y for _, y, _ in tr]); counts = np.bincount(ys, minlength=2)
    w = None
    if cfg["class_weight"]:
        w = torch.tensor(len(ys) / (2 * np.maximum(counts, 1)), dtype=torch.float).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg["epochs"], 1))
    dl = make_loader(tr, True, cfg["augment"], cfg["batch_size"], cfg["num_workers"])
    best, best_state, bad, log = -float("inf"), None, 0, []
    for ep in range(cfg["epochs"]):
        model.train(); tot, n = 0.0, 0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = loss_fn(model(x), y); loss.backward(); opt.step()
            tot += loss.item() * len(y); n += len(y)
        sched.step()
        lg, yv = get_logits(model, va, device, cfg["batch_size"], cfg["num_workers"])
        vloss = F.cross_entropy(torch.tensor(lg), torch.tensor(yv)).item()
        auc = safe_auc(yv, softmax(lg)[:, 1])
        score = auc if np.isfinite(auc) else -vloss
        log.append({"epoch": ep, "train_loss": tot / max(n, 1), "val_loss": vloss, "val_auroc": auc})
        print(f"  ep{ep} train_loss={tot/max(n,1):.4f} val_loss={vloss:.4f} val_auroc={auc:.4f}")
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print("  early stopping"); break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, log


def _load_synthetic(synthetic_dir, cls_name, ratio, real_train_items):
    """Return synthetic items to append, limited by ratio of minority-class count.

    Parameters
    ----------
    synthetic_dir : str
        Directory containing synthetic images for ``cls_name``.
    cls_name : str
        Class name (one of CLASSES).
    ratio : float
        Fraction of the minority-class real count to add (0.5 = 50%, 1.0 = 100%).
    real_train_items : list
        The real training items (used to compute minority-class count).

    Returns
    -------
    list of (path, label, patient, source) 4-tuples
    """
    ys = np.array([y for _, y, _ in real_train_items])
    counts = np.bincount(ys, minlength=2)
    minority_count = int(counts.min())
    max_add = max(1, round(minority_count * ratio))

    all_syn = collect_synthetic(synthetic_dir, cls_name)
    return all_syn[:max_add]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train pneumonia classifiers (own models, no APIs)")
    ap.add_argument("--internal", required=True, help="folder with NORMAL/ PNEUMONIA/ (e.g. Kermany)")
    ap.add_argument("--external", help="external test folder (same structure)")
    ap.add_argument("--models", nargs="+", default=["resnet50"], choices=ARCHS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--no-augment", action="store_true", help="ablation")
    ap.add_argument("--no-class-weight", action="store_true", help="ablation")
    ap.add_argument("--no-pretrained", action="store_true", help="ablation")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--out", default="results")
    # ── GAN augmentation (optional) ──────────────────────────────────────
    ap.add_argument(
        "--synthetic-dir",
        default=None,
        help=(
            "Directory of synthetic images produced by "
            "`python -m pneumreliab.gan sample`. "
            "Images are added to training only; val/test/external remain real."
        ),
    )
    ap.add_argument(
        "--synthetic-ratio",
        type=float,
        default=1.0,
        help=(
            "Fraction of minority-class real count to add from synthetic-dir "
            "(e.g. 0.5 = 50%%, 1.0 = 100%%). Default 1.0."
        ),
    )
    ap.add_argument(
        "--synthetic-cls",
        default="NORMAL",
        choices=CLASSES,
        help="Which class the synthetic images belong to (default: NORMAL).",
    )
    a = ap.parse_args(argv)

    device = get_device(); items = collect(a.internal)
    print(f"internal images={len(items)} device={device}")
    for arch in a.models:
        for seed in a.seeds:
            run = os.path.join(a.out, f"{a.tag}_{arch}_s{seed}")
            print(f"== {run}")
            set_seed(seed)
            sp = patient_split(items, seed)
            save_manifest(sp, os.path.join(run, "split.csv"))

            train_items = sp["train"]
            n_real = {cls: 0 for cls in CLASSES}
            for _, l, _ in train_items:
                n_real[CLASSES[l]] += 1

            n_synthetic = {cls: 0 for cls in CLASSES}
            if a.synthetic_dir:
                syn_items = _load_synthetic(
                    a.synthetic_dir, a.synthetic_cls, a.synthetic_ratio, train_items
                )
                # Guard: no synthetic paths may leak into val / test / external.
                # External items are not loaded yet at this point; the check is
                # conservative (paths from syn_items vs sp["val"] + sp["test"]).
                train_items = append_synthetic_to_train(
                    train_items, syn_items,
                    val_items=sp["val"],
                    test_items=sp["test"],
                    external_items=None,   # external not in splits dict
                )
                for item in syn_items:
                    n_synthetic[CLASSES[item[1]]] += 1
                print(
                    f"  [synthetic] added {len(syn_items)} images "
                    f"({a.synthetic_cls}, ratio={a.synthetic_ratio})"
                )

            cfg = {
                "arch": arch, "seed": seed, "epochs": a.epochs,
                "batch_size": a.batch_size,
                "lr": a.lr or (1e-3 if arch == "smallcnn" else 1e-4),
                "patience": a.patience,
                "augment": not a.no_augment, "class_weight": not a.no_class_weight,
                "pretrained": not a.no_pretrained, "num_workers": a.num_workers,
                # GAN augmentation provenance
                "synthetic_dir": a.synthetic_dir,
                "synthetic_ratio": a.synthetic_ratio if a.synthetic_dir else None,
                "synthetic_cls": a.synthetic_cls if a.synthetic_dir else None,
                "n_real_train": n_real,
                "n_synthetic_train": n_synthetic,
            }
            model = build(arch, pretrained=cfg["pretrained"])
            model, log = train_model(model, train_items, sp["val"], cfg, device)

            # Calibration fitted on REAL val images only (unchanged)
            lv, yv = get_logits(model, sp["val"], device, num_workers=a.num_workers)
            T = fit_temperature(lv, yv)

            torch.save({"arch": arch, "state_dict": model.state_dict(), "T": T, "config": cfg},
                       os.path.join(run, "model.pt"))
            save_json({**cfg, "T": T, "env": env_info(),
                       "n": {k: len(v) for k, v in sp.items()}},
                      os.path.join(run, "config.json"))
            save_json(log, os.path.join(run, "train_log.json"))
            evaluate_run(run, a.external, device, a.num_workers)


if __name__ == "__main__":
    main()
