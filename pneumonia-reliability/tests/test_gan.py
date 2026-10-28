"""Tests for GAN-based data augmentation.

Tests
-----
1. assert_no_synthetic_outside_train raises AssertionError when a synthetic
   path appears in val, test, or external.
2. A GAN trains for 1 epoch on the fixture dataset and samples images of the
   correct shape (224x224, grayscale PNG).
3. Old manifest format (no 'source' column) still loads correctly via
   load_manifest.
4. load_manifest_full returns the 'source' field (defaulting to 'real' for
   old manifests).
"""
import os
import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pneumreliab.data import (
    CLASSES,
    SOURCE_REAL,
    SOURCE_SYNTHETIC,
    assert_no_synthetic_outside_train,
    collect_synthetic,
    append_synthetic_to_train,
    load_manifest,
    load_manifest_full,
    save_manifest,
    patient_split,
    collect,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_synthetic_dir(tmp_path):
    """Create a flat directory of tiny synthetic NORMAL PNG files."""
    d = tmp_path / "synthetic" / "NORMAL"
    d.mkdir(parents=True)
    rng = np.random.default_rng(42)
    for i in range(5):
        arr = rng.integers(0, 255, (224, 224), dtype=np.uint8)
        Image.fromarray(arr, mode="L").save(d / f"synth_NORMAL_{i:06d}.png")
    return str(d)


@pytest.fixture
def run_dir_with_split(tmp_path, synthetic_root):
    """Create a results/<run> directory with split.csv from the synthetic_root fixture."""
    run = tmp_path / "results" / "gan_test_smallcnn_s0"
    run.mkdir(parents=True)
    items = collect(synthetic_root)
    sp = patient_split(items, seed=0)
    save_manifest(sp, str(run / "split.csv"))
    return str(run), sp


# ──────────────────────────────────────────────────────────────────────────────
# 1. Leakage assertion
# ──────────────────────────────────────────────────────────────────────────────

def test_synthetic_in_val_raises(tmp_path):
    """A synthetic path in val must raise AssertionError."""
    syn_path = str(tmp_path / "synth_NORMAL_000000.png")
    # Touch the file so the path exists (not strictly required for the assertion)
    Path(syn_path).touch()

    syn_paths = {syn_path}
    val_items = [(syn_path, 0, "pid0")]   # synthetic path leaking into val
    test_items = []

    with pytest.raises(AssertionError, match="LEAKAGE"):
        assert_no_synthetic_outside_train(syn_paths, val_items, test_items)


def test_synthetic_in_test_raises(tmp_path):
    """A synthetic path in test must raise AssertionError."""
    syn_path = str(tmp_path / "synth_NORMAL_000000.png")
    Path(syn_path).touch()

    syn_paths = {syn_path}
    val_items = []
    test_items = [(syn_path, 0, "pid0")]   # synthetic path leaking into test

    with pytest.raises(AssertionError, match="LEAKAGE"):
        assert_no_synthetic_outside_train(syn_paths, val_items, test_items)


def test_synthetic_in_external_raises(tmp_path):
    """A synthetic path in external evaluation must raise AssertionError."""
    syn_path = str(tmp_path / "synth_NORMAL_000000.png")
    Path(syn_path).touch()

    syn_paths = {syn_path}
    external_items = [(syn_path, 0, "pid0")]

    with pytest.raises(AssertionError, match="LEAKAGE"):
        assert_no_synthetic_outside_train(syn_paths, [], [], external_items)


def test_synthetic_in_train_only_passes(tmp_path, tiny_synthetic_dir):
    """Synthetic paths that appear only in train should not raise."""
    syn_items = collect_synthetic(tiny_synthetic_dir, "NORMAL")
    syn_paths = {item[0] for item in syn_items}
    # val and test have distinct (real) paths
    val_items = [("/real/NORMAL/img_val.png", 0, "pv")]
    test_items = [("/real/NORMAL/img_test.png", 0, "pt")]
    # Must not raise
    assert_no_synthetic_outside_train(syn_paths, val_items, test_items)


def test_append_synthetic_to_train_returns_combined(tmp_path, tiny_synthetic_dir):
    """append_synthetic_to_train should combine real + synthetic triples."""
    real_train = [("/real/NORMAL/a.png", 0, "p1"), ("/real/PNEUMONIA/b.png", 1, "p2")]
    syn_items = collect_synthetic(tiny_synthetic_dir, "NORMAL")
    combined = append_synthetic_to_train(real_train, syn_items)
    assert len(combined) == len(real_train) + len(syn_items)
    # All items are 3-tuples
    for item in combined:
        assert len(item) == 3


# ──────────────────────────────────────────────────────────────────────────────
# 2. GAN smoke test: train 1 epoch, sample correct shape
# ──────────────────────────────────────────────────────────────────────────────

def test_gan_train_one_epoch_and_sample(run_dir_with_split, tmp_path):
    """GAN trains for 1 epoch and sampled images are 224x224."""
    from pneumreliab.gan import train_gan, sample_images

    run_dir, _ = run_dir_with_split
    out_dir = str(tmp_path / "synthetic" / "NORMAL")

    # 1-epoch smoke test; batch_size small so it works on CPU in CI
    train_gan(
        run_dir=run_dir,
        epochs=1,
        batch_size=4,
        seed=42,
        sample_every=1,   # always emit a grid
        num_workers=0,
    )

    # Generator should be saved
    assert os.path.exists(os.path.join(run_dir, "generator.pt"))
    assert os.path.exists(os.path.join(run_dir, "gan_loss.json"))

    # Sample 3 images
    sample_images(run_dir, "NORMAL", n=3, out_dir=out_dir, seed=0, upsample_size=224)

    pngs = sorted(Path(out_dir).glob("*.png"))
    assert len(pngs) == 3, f"Expected 3 PNGs, got {len(pngs)}"

    for png in pngs:
        img = Image.open(png)
        assert img.size == (224, 224), f"Expected 224x224, got {img.size}"
        assert img.mode == "L", f"Expected grayscale (L), got {img.mode}"


# ──────────────────────────────────────────────────────────────────────────────
# 3. Old manifest (no 'source' column) still loads
# ──────────────────────────────────────────────────────────────────────────────

def test_old_manifest_loads(tmp_path):
    """Manifests without the 'source' column must still load via load_manifest."""
    old_path = str(tmp_path / "old_split.csv")
    with open(old_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "path", "label", "patient"])   # no 'source' column
        w.writerow(["train", "/data/NORMAL/a.png", "0", "p1"])
        w.writerow(["val",   "/data/NORMAL/b.png", "0", "p2"])
        w.writerow(["test",  "/data/PNEUMONIA/c.png", "1", "p3"])

    splits = load_manifest(old_path)
    assert len(splits["train"]) == 1
    assert len(splits["val"]) == 1
    assert len(splits["test"]) == 1
    # Returns 3-tuples (path, label, patient) — source dropped
    p, l, pid = splits["train"][0]
    assert p == "/data/NORMAL/a.png"
    assert l == 0
    assert pid == "p1"


def test_old_manifest_full_defaults_source_to_real(tmp_path):
    """load_manifest_full on an old manifest should default source to 'real'."""
    old_path = str(tmp_path / "old_split.csv")
    with open(old_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "path", "label", "patient"])
        w.writerow(["train", "/data/NORMAL/a.png", "0", "p1"])

    splits = load_manifest_full(old_path)
    _, _, _, source = splits["train"][0]
    assert source == SOURCE_REAL


def test_new_manifest_saves_and_loads_source(tmp_path, synthetic_root):
    """save_manifest writes 'source' column; load_manifest_full reads it."""
    items = collect(synthetic_root)
    sp = patient_split(items, seed=1)
    manifest_path = str(tmp_path / "split.csv")
    save_manifest(sp, manifest_path)

    full = load_manifest_full(manifest_path)
    for split_name, rows in full.items():
        for _, _, _, source in rows:
            assert source == SOURCE_REAL, (
                f"Expected all sources to be '{SOURCE_REAL}' in a real manifest, "
                f"got '{source}' in split '{split_name}'"
            )
