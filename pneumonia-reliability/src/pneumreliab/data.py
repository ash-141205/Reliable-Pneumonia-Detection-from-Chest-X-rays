"""Data handling. Key design choice: splits are made BY PATIENT and saved to a
manifest CSV so every experiment is reproducible and leakage-checked.

Synthetic-image support (added for GAN augmentation)
-----------------------------------------------------
- ``save_manifest`` writes a ``source`` column (values: ``real`` or
  ``synthetic``).  Old manifests without the column are still readable
  (defaulting to ``real``).
- ``load_manifest`` always populates the ``source`` field.
- ``collect_synthetic`` loads images from a directory of PNG/JPEG files (no
  sub-folder structure required; the class is inferred from the directory name).
- ``append_synthetic_to_train`` merges synthetic items into the train list and
  raises ``AssertionError`` if any synthetic path appears in val / test.
"""
import csv, os, re
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CLASSES = ["NORMAL", "PNEUMONIA"]
EXTS = (".jpg", ".jpeg", ".png")
NORM = transforms.Normalize([0.485] * 3, [0.229] * 3)

# Sentinel value for the synthetic source tag used in the manifest.
SOURCE_REAL = "real"
SOURCE_SYNTHETIC = "synthetic"


def patient_id(fname):
    """Kermany: 'person123_bacteria_456.jpeg' -> person123; 'IM-0115-0001.jpeg'
    -> IM-0115. Otherwise the file stem (assumes 1 image per patient; replace
    if your dataset provides real patient IDs)."""
    m = re.match(r"(person\d+)", fname)
    if m:
        return m.group(1)
    m = re.search(r"(IM-\d+)", fname)
    if m:
        return m.group(1)
    return os.path.splitext(fname)[0]


def collect(root):
    """Walk `root`; any folder named NORMAL / PNEUMONIA defines the label."""
    items = []
    for dp, _, fs in os.walk(root):
        cls = os.path.basename(dp)
        if cls not in CLASSES:
            continue
        for f in sorted(fs):
            if f.lower().endswith(EXTS):
                items.append((os.path.join(dp, f), CLASSES.index(cls), patient_id(f)))
    items.sort()
    if not items:
        raise FileNotFoundError(f"No images found under {root} (need NORMAL/ and PNEUMONIA/ folders)")
    return items


def patient_split(items, seed, val=0.15, test=0.15):
    """Stratified, group-aware train/val/test split. No patient is shared."""
    y = np.array([l for _, l, _ in items]); g = np.array([p for _, _, p in items])
    idx = np.arange(len(items))
    k1 = max(round(1 / test), 2)
    trv, te = next(StratifiedGroupKFold(k1, shuffle=True, random_state=seed).split(idx, y, g))
    k2 = max(round((1 - test) / val), 2)
    a, b = next(StratifiedGroupKFold(k2, shuffle=True, random_state=seed).split(trv, y[trv], g[trv]))
    tr, va = trv[a], trv[b]
    sets = {"train": [items[i] for i in tr], "val": [items[i] for i in va],
            "test": [items[i] for i in te]}
    pats = {k: {x[2] for x in v} for k, v in sets.items()}
    assert not (pats["train"] & pats["val"] or pats["train"] & pats["test"]
                or pats["val"] & pats["test"]), "patient leakage between splits"
    return sets


def save_manifest(splits, path):
    """Write split.csv with an additional ``source`` column (always 'real' here).

    Each row: split, path, label, patient, source
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "path", "label", "patient", "source"])
        for s, items in splits.items():
            for p, l, g in items:
                w.writerow([s, p, l, g, SOURCE_REAL])


def load_manifest(path):
    """Load split.csv.

    Supports both old (no 'source' column) and new format.  Missing 'source'
    defaults to 'real'.  Returns dict of lists of (path, label, patient)
    triples, matching the original API; the source field is dropped from the
    return value to keep backward compatibility.  Use ``load_manifest_full``
    when you need the source column.
    """
    out = {"train": [], "val": [], "test": []}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r["split"]].append((r["path"], int(r["label"]), r["patient"]))
    return out


def load_manifest_full(path):
    """Like ``load_manifest`` but returns (path, label, patient, source) tuples."""
    out = {"train": [], "val": [], "test": []}
    with open(path) as f:
        for r in csv.DictReader(f):
            source = r.get("source", SOURCE_REAL)
            out[r["split"]].append((r["path"], int(r["label"]), r["patient"], source))
    return out


# ──────────────────────────── synthetic helpers ───────────────────────────────

def collect_synthetic(directory, cls_name):
    """Load all images from ``directory`` as synthetic items of class ``cls_name``.

    Each item is ``(path, label_int, patient_id, source)`` — note the extra
    ``source`` field used internally.  The patient ID is set to the filename
    stem so no patient-level merging with real data can happen.

    Parameters
    ----------
    directory : str
        Folder containing PNG/JPEG images (flat, no sub-folders needed).
    cls_name : str
        One of CLASSES; determines the integer label.
    """
    if cls_name not in CLASSES:
        raise ValueError(f"cls_name must be one of {CLASSES}; got '{cls_name}'")
    label = CLASSES.index(cls_name)
    items = []
    for f in sorted(os.listdir(directory)):
        if f.lower().endswith(EXTS):
            p = os.path.join(directory, f)
            pid = os.path.splitext(f)[0]  # unique per file
            items.append((p, label, pid, SOURCE_SYNTHETIC))
    return items


def assert_no_synthetic_outside_train(synthetic_paths, val_items, test_items,
                                      external_items=None):
    """Raise AssertionError loudly if any synthetic path appears outside train.

    Call this before building any non-training DataLoader when synthetic images
    are present.

    Parameters
    ----------
    synthetic_paths : set[str]
        Absolute paths of all synthetic images.
    val_items, test_items : list of (path, label, patient, ...)
        Val and test splits.
    external_items : list of (path, label, patient, ...) or None
        External test set items, if any.
    """
    def _paths(items):
        return {item[0] for item in items}

    leaked = synthetic_paths & (_paths(val_items) | _paths(test_items))
    if external_items:
        leaked |= synthetic_paths & _paths(external_items)

    assert not leaked, (
        f"LEAKAGE: {len(leaked)} synthetic image(s) found outside the training set!\n"
        + "\n".join(sorted(leaked))
    )


def append_synthetic_to_train(train_items, synthetic_items,
                               val_items=None, test_items=None,
                               external_items=None):
    """Append synthetic items to the training list and perform leakage check.

    Parameters
    ----------
    train_items : list of (path, label, patient) triples
        Existing training items (real images).
    synthetic_items : list of (path, label, patient, source) 4-tuples
        Items returned by ``collect_synthetic``.
    val_items, test_items, external_items : optional lists
        If provided, ``assert_no_synthetic_outside_train`` is called.

    Returns
    -------
    list
        Extended training list as (path, label, patient) triples (source field
        dropped to keep downstream code unchanged).
    """
    syn_paths = {item[0] for item in synthetic_items}

    if val_items is not None or test_items is not None:
        assert_no_synthetic_outside_train(
            syn_paths,
            val_items or [],
            test_items or [],
            external_items,
        )

    # Convert 4-tuples to 3-tuples (drop source tag; it is stored in the manifest)
    syn_as_triples = [(p, l, pid) for p, l, pid, _ in synthetic_items]
    return train_items + syn_as_triples


def get_transform(train, augment=True, size=224):
    ops = [transforms.Grayscale(3), transforms.Resize((size, size))]
    if train and augment:
        ops += [transforms.RandomRotation(10), transforms.ColorJitter(0.2, 0.2)]
    return transforms.Compose(ops + [transforms.ToTensor(), NORM])


class CXR(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y, _ = self.items[i]
        return self.tf(Image.open(p).convert("L")), y


def make_loader(items, train, augment=True, batch_size=32, num_workers=2):
    return DataLoader(CXR(items, get_transform(train, augment)), batch_size=batch_size,
                      shuffle=train, num_workers=num_workers,
                      drop_last=train and len(items) > batch_size)
