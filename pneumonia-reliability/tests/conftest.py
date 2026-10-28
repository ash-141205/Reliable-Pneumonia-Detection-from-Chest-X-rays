import os, sys
import numpy as np
import pytest
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def synthetic_root(tmp_path):
    """Tiny fake dataset in Kermany layout: 2 images per patient."""
    rng = np.random.default_rng(0)
    for cls, mean in (("NORMAL", 90), ("PNEUMONIA", 150)):
        d = tmp_path / "data" / cls; d.mkdir(parents=True)
        for i in range(30):
            for j in (1, 2):
                name = f"IM-{1000+i}-000{j}.jpeg" if cls == "NORMAL" else f"person{i}_bacteria_{i*10+j}.jpeg"
                arr = np.clip(rng.normal(mean, 30, (64, 64)), 0, 255).astype(np.uint8)
                Image.fromarray(arr).save(d / name)
    return str(tmp_path / "data")
