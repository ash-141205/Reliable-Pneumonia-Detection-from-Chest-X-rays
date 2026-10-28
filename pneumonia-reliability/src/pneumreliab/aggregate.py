"""Collect metrics.json from all runs -> mean +/- std over seeds (summary.md/csv)."""
import argparse, csv, glob, os, re
from collections import defaultdict
import numpy as np
from .utils import load_json

KEYS = ["auroc", "auprc", "sensitivity", "specificity", "f1", "accuracy", "brier", "ece"]


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results")
    a = ap.parse_args(argv)
    groups = defaultdict(list)
    for f in glob.glob(os.path.join(a.results, "*", "metrics.json")):
        groups[re.sub(r"_s\d+$", "", os.path.basename(os.path.dirname(f)))].append(load_json(f))
    rows = []
    for g, runs in sorted(groups.items()):
        for split in ("internal_test", "external_test"):
            for cal in ("raw", "calibrated"):
                vals = [r[split][cal] for r in runs if split in r]
                if not vals:
                    continue
                row = {"group": g, "split": split, "calibration": cal, "seeds": len(vals)}
                for k in KEYS:
                    v = np.array([x[k] for x in vals], dtype=float)
                    row[k] = f"{np.nanmean(v):.3f} ± {np.nanstd(v):.3f}"
                rows.append(row)
    if not rows:
        print("no metrics.json found"); return
    hdr = list(rows[0])
    with open(os.path.join(a.results, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, hdr); w.writeheader(); w.writerows(rows)
    md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    md += ["| " + " | ".join(str(r[h]) for h in hdr) + " |" for r in rows]
    open(os.path.join(a.results, "summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
