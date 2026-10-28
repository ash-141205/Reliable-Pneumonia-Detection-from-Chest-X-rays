import json, os, platform, random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def env_info():
    import sklearn, torchvision
    return {"python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "sklearn": sklearn.__version__,
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
