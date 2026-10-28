import os
from pneumreliab import train, aggregate, shortcut


def test_end_to_end_tiny_run(synthetic_root, tmp_path):
    out = str(tmp_path / "res")
    train.main(["--internal", synthetic_root, "--external", synthetic_root, "--models", "smallcnn",
                "--seeds", "0", "--epochs", "1", "--batch-size", "8", "--num-workers", "0",
                "--out", out])
    run = os.path.join(out, "main_smallcnn_s0")
    for f in ("model.pt", "split.csv", "config.json", "train_log.json", "metrics.json", "plots.png"):
        assert os.path.exists(os.path.join(run, f)), f
    aggregate.main(["--results", out])
    assert os.path.exists(os.path.join(out, "summary.md"))
    res = shortcut.shortcut_audit(shortcut.collect(synthetic_root), shortcut.collect(synthetic_root), n=20)
    assert 0 <= res["border_only_source_auroc"] <= 1
