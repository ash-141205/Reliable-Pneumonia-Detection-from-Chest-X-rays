import numpy as np
import torch
from PIL import Image
from .models import target_layer


def gradcam(model, x, cls, layer=None):
    """Grad-CAM heatmap in [0,1] for class `cls` of a (1,3,H,W) tensor."""
    layer = layer or target_layer(model); feats, grads = [], []

    def fwd(_, __, o):
        feats.append(o); o.register_hook(lambda g: grads.append(g))
    h = layer.register_forward_hook(fwd)
    with torch.enable_grad():
        model.zero_grad(); out = model(x); out[0, cls].backward()
    h.remove()
    w = grads[0].mean((2, 3), keepdim=True)
    cam = torch.relu((w * feats[0]).sum(1))[0].detach().cpu().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def overlay(img, cam, alpha=0.45, size=224):
    from matplotlib import pyplot as plt
    c = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((size, size))) / 255
    heat = plt.get_cmap("jet")(c)[..., :3]
    base = np.asarray(img.resize((size, size)).convert("RGB")) / 255
    return (1 - alpha) * base + alpha * heat
