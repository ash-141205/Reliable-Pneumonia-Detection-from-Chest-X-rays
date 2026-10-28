"""Class-conditional DCGAN for GAN-based augmentation of chest X-rays.

Design choices
--------------
- 64x64 grayscale output (upsampled to 224 at sample time).
- Class embedding concatenated to the noise vector in the generator, and as
  a one-hot appended to every discriminator feature map (AC-GAN style light).
- Stabilisers: label smoothing on the real labels, instance noise on
  discriminator inputs (annealed), spectral normalisation on all D conv layers.
- Optimiser: Adam(lr=2e-4, betas=(0.5, 0.999)) for both G and D.
- Fixed-seed noise grid saved every --sample-every epochs.
- No pretrained weights, no network calls.

CLI
---
  python -m pneumreliab.gan train  --run-dir results/main_resnet50_s0 --epochs 50 --seed 0
  python -m pneumreliab.gan sample --run-dir results/main_resnet50_s0 --cls NORMAL --n 500 \\
                                    --out results/main_resnet50_s0/synthetic/NORMAL
"""

from __future__ import annotations
import argparse, json, math, os, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .data import CLASSES, load_manifest
from .utils import get_device, save_json, set_seed

# ──────────────────────────── constants ──────────────────────────────────────
LATENT_DIM = 128
IMG_SIZE = 64          # GAN native resolution
N_CLASSES = len(CLASSES)   # 2: NORMAL=0, PNEUMONIA=1

# ──────────────────────────── helpers ────────────────────────────────────────

def _spectral(layer):
    """Apply spectral normalisation if it is a Conv/Linear layer."""
    return nn.utils.spectral_norm(layer)


# ──────────────────────────── Generator ──────────────────────────────────────

class Generator(nn.Module):
    """Conditional DCGAN generator.  z in R^LATENT_DIM, cls in {0,...,N_CLASSES-1}."""

    def __init__(self, latent_dim=LATENT_DIM, n_classes=N_CLASSES):
        super().__init__()
        self.embed = nn.Embedding(n_classes, n_classes)
        in_ch = latent_dim + n_classes

        def blk(i, o, k=4, s=2, p=1, bn=True):
            layers = [nn.ConvTranspose2d(i, o, k, s, p, bias=not bn)]
            if bn:
                layers.append(nn.BatchNorm2d(o))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        # 1x1 -> 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            blk(512, 256),   # 8x8
            blk(256, 128),   # 16x16
            blk(128, 64),    # 32x32
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=True),   # 64x64, 1 channel
            nn.Tanh(),
        )

    def forward(self, z, cls):
        emb = self.embed(cls)                    # (B, n_classes)
        x = torch.cat([z, emb], dim=1)           # (B, latent+n_classes)
        x = x.unsqueeze(-1).unsqueeze(-1)        # (B, C, 1, 1)
        return self.net(x)


# ──────────────────────────── Discriminator ──────────────────────────────────

class Discriminator(nn.Module):
    """Conditional DCGAN discriminator with spectral norm on convolutions."""

    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        self.embed = nn.Embedding(n_classes, IMG_SIZE * IMG_SIZE)

        def blk(i, o, k=4, s=2, p=1):
            return nn.Sequential(
                _spectral(nn.Conv2d(i, o, k, s, p, bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
            )

        # input: (B, 2, 64, 64) -- image channel + class channel
        self.net = nn.Sequential(
            blk(2, 64),          # 32x32
            blk(64, 128),        # 16x16
            blk(128, 256),       # 8x8
            blk(256, 512),       # 4x4
            _spectral(nn.Conv2d(512, 1, 4, 1, 0, bias=True)),   # 1x1
        )

    def forward(self, img, cls):
        # Class conditioning: broadcast a class-channel over the image spatial dims
        emb = self.embed(cls).view(cls.size(0), 1, IMG_SIZE, IMG_SIZE)
        x = torch.cat([img, emb], dim=1)         # (B, 2, 64, 64)
        return self.net(x).view(-1)


# ──────────────────────────── Dataset ────────────────────────────────────────

class _GrayscaleDataset(Dataset):
    """Loads 64x64 grayscale images from a manifest train split."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label, _ = self.items[i]
        img = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img, dtype=np.float32) / 127.5 - 1.0   # [-1, 1]
        return torch.tensor(arr).unsqueeze(0), label


# ──────────────────────────── FID (optional) ─────────────────────────────────

def _try_fid(real_dir, fake_dir, device):
    """Return FID if torch_fidelity or pytorch_fid is installed, else None."""
    # Try torch-fidelity first
    try:
        import torch_fidelity
        res = torch_fidelity.calculate_metrics(
            input1=real_dir, input2=fake_dir,
            cuda=(device == "cuda"), fid=True, verbose=False
        )
        return float(res["frechet_inception_distance"])
    except ImportError:
        pass
    # Try pytorch-fid
    try:
        from pytorch_fid import fid_score
        fid_val = fid_score.calculate_fid_given_paths(
            [real_dir, fake_dir], batch_size=50,
            device=device, dims=2048, num_workers=0
        )
        return float(fid_val)
    except ImportError:
        pass
    return None


# ──────────────────────────── instance noise ─────────────────────────────────

def _add_instance_noise(x, sigma):
    if sigma <= 0:
        return x
    return x + sigma * torch.randn_like(x)


# ──────────────────────────── Training ───────────────────────────────────────

def train_gan(
    run_dir,
    epochs=50,
    batch_size=64,
    seed=0,
    sample_every=5,
    noise_start=0.1,
    label_smooth=0.1,
    num_workers=2,
):
    """Train a class-conditional DCGAN and save artifacts to run_dir."""
    device = get_device()
    set_seed(seed)
    print(f"[GAN] device={device} epochs={epochs} batch_size={batch_size} seed={seed}")

    # ── load train split only ──────────────────────────────────────────────
    manifest_path = os.path.join(run_dir, "split.csv")
    if not os.path.exists(manifest_path):
        sys.exit(f"[GAN] split.csv not found in {run_dir}. Run training first.")
    splits = load_manifest(manifest_path)
    train_items = splits["train"]

    # Safety: assert no synthetic paths slip into GAN training
    for p, _, _ in train_items:
        assert "synthetic" not in p.replace("\\", "/").split("/"), \
            f"[GAN] synthetic image found in train split: {p}"

    print(f"[GAN] train items: {len(train_items)}")
    counts = [0, 0]
    for _, l, _ in train_items:
        counts[l] += 1
    print(f"[GAN] class counts: NORMAL={counts[0]} PNEUMONIA={counts[1]}")

    dataset = _GrayscaleDataset(train_items)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=len(dataset) > batch_size,
        pin_memory=(device == "cuda")
    )

    # ── models ────────────────────────────────────────────────────────────
    G = Generator().to(device)
    D = Discriminator().to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # Fixed noise for visual monitoring
    fixed_noise = torch.randn(N_CLASSES * 8, LATENT_DIM, device=device)
    fixed_cls = torch.tensor(
        [c for c in range(N_CLASSES) for _ in range(8)], device=device
    )

    loss_history = []
    out_dir = Path(run_dir)

    for ep in range(epochs):
        G.train(); D.train()
        g_losses, d_losses = [], []

        # Instance noise annealed to 0 over epochs
        noise_sigma = noise_start * max(0.0, 1.0 - ep / max(epochs, 1))

        for real_imgs, real_cls in loader:
            real_imgs = real_imgs.to(device)
            real_cls = real_cls.to(device)
            bs = real_imgs.size(0)

            # ── Discriminator step ────────────────────────────────────────
            opt_D.zero_grad()
            noisy_real = _add_instance_noise(real_imgs, noise_sigma)
            real_smooth = 1.0 - label_smooth       # e.g. 0.9
            d_real = D(noisy_real, real_cls)
            loss_D_real = F.binary_cross_entropy_with_logits(
                d_real, torch.full_like(d_real, real_smooth)
            )

            z = torch.randn(bs, LATENT_DIM, device=device)
            fake_cls = torch.randint(0, N_CLASSES, (bs,), device=device)
            fake_imgs = G(z, fake_cls).detach()
            noisy_fake = _add_instance_noise(fake_imgs, noise_sigma)
            d_fake = D(noisy_fake, fake_cls)
            loss_D_fake = F.binary_cross_entropy_with_logits(
                d_fake, torch.zeros_like(d_fake)
            )
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward(); opt_D.step()

            # ── Generator step ────────────────────────────────────────────
            opt_G.zero_grad()
            z2 = torch.randn(bs, LATENT_DIM, device=device)
            fake_cls2 = torch.randint(0, N_CLASSES, (bs,), device=device)
            fake_imgs2 = G(z2, fake_cls2)
            noisy_fake2 = _add_instance_noise(fake_imgs2, noise_sigma)
            d_out = D(noisy_fake2, fake_cls2)
            loss_G = F.binary_cross_entropy_with_logits(
                d_out, torch.ones_like(d_out)   # G wants D to say "real"
            )
            loss_G.backward(); opt_G.step()

            g_losses.append(loss_G.item())
            d_losses.append(loss_D.item())

        mean_g = float(np.mean(g_losses)) if g_losses else float("nan")
        mean_d = float(np.mean(d_losses)) if d_losses else float("nan")
        loss_history.append({"epoch": ep, "g_loss": mean_g, "d_loss": mean_d})
        print(
            f"  [GAN] ep{ep:03d} g_loss={mean_g:.4f} d_loss={mean_d:.4f}"
            f" noise_sigma={noise_sigma:.4f}"
        )

        # ── sample grid ───────────────────────────────────────────────────
        if (ep + 1) % sample_every == 0 or ep == epochs - 1:
            _save_sample_grid(
                G, fixed_noise, fixed_cls, device,
                str(out_dir / f"gan_grid_ep{ep+1:04d}.png")
            )

    # ── save generator ────────────────────────────────────────────────────
    torch.save(
        {"state_dict": G.state_dict(), "latent_dim": LATENT_DIM,
         "n_classes": N_CLASSES, "seed": seed},
        str(out_dir / "generator.pt")
    )
    print(f"[GAN] generator saved -> {out_dir / 'generator.pt'}")

    # ── loss curve ────────────────────────────────────────────────────────
    save_json(loss_history, str(out_dir / "gan_loss.json"))
    _plot_loss(loss_history, str(out_dir / "gan_loss.png"))

    return G


def _save_sample_grid(G, fixed_noise, fixed_cls, device, path):
    G.eval()
    with torch.no_grad():
        imgs = G(fixed_noise, fixed_cls).cpu()   # (N, 1, 64, 64) in [-1,1]
    imgs = (imgs * 0.5 + 0.5).clamp(0, 1)       # -> [0, 1]
    n = imgs.size(0)
    cols = 8; rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
    ax_flat = np.array(axes).ravel()
    for i, ax in enumerate(ax_flat):
        if i < n:
            ax.imshow(imgs[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_title(CLASSES[fixed_cls[i].item()], fontsize=6)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    G.train()


def _plot_loss(history, path):
    eps = [r["epoch"] for r in history]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(eps, [r["g_loss"] for r in history], label="G loss")
    ax.plot(eps, [r["d_loss"] for r in history], label="D loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("BCE loss")
    ax.set_title("GAN training loss"); ax.legend(); fig.tight_layout()
    fig.savefig(path, dpi=120); plt.close(fig)


# ──────────────────────────── Sampling ───────────────────────────────────────

def sample_images(run_dir, cls_name, n, out_dir, seed=0, upsample_size=224):
    """Load generator.pt and write n synthetic images for class cls_name."""
    if cls_name not in CLASSES:
        sys.exit(f"[GAN] --cls must be one of {CLASSES}; got '{cls_name}'")
    cls_idx = CLASSES.index(cls_name)
    device = get_device()
    set_seed(seed)

    ck = torch.load(os.path.join(run_dir, "generator.pt"), map_location=device)
    G = Generator(ck["latent_dim"], ck["n_classes"]).to(device)
    G.load_state_dict(ck["state_dict"]); G.eval()

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cls_t = torch.tensor([cls_idx], device=device)

    with torch.no_grad():
        for i in range(n):
            z = torch.randn(1, ck["latent_dim"], device=device)
            img_t = G(z, cls_t)                            # (1,1,64,64) [-1,1]
            arr = (
                (img_t[0, 0].cpu().numpy() * 0.5 + 0.5) * 255
            ).clip(0, 255).astype(np.uint8)
            pil = Image.fromarray(arr, mode="L").resize(
                (upsample_size, upsample_size), Image.BICUBIC
            )
            fname = os.path.join(out_dir, f"synth_{cls_name}_{i:06d}.png")
            pil.save(fname)

    print(f"[GAN] wrote {n} synthetic '{cls_name}' images -> {out_dir}")


# ──────────────────────────── CLI ────────────────────────────────────────────

def _cli_train(argv=None):
    ap = argparse.ArgumentParser(description="Train GAN on a run's train split")
    ap.add_argument("--run-dir", required=True,
                    help="results/<tag>_<arch>_s<seed> directory")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-every", type=int, default=5,
                    help="save a sample grid every N epochs (default 5)")
    ap.add_argument("--num-workers", type=int, default=2)
    a = ap.parse_args(argv)
    train_gan(
        a.run_dir, a.epochs, a.batch_size, a.seed,
        a.sample_every, num_workers=a.num_workers
    )


def _cli_sample(argv=None):
    ap = argparse.ArgumentParser(description="Sample images from a trained GAN")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--cls", required=True, choices=CLASSES,
                    help="class to generate")
    ap.add_argument("--n", type=int, required=True,
                    help="number of images to generate")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=224,
                    help="upsample size in pixels (default 224)")
    a = ap.parse_args(argv)
    sample_images(a.run_dir, a.cls, a.n, a.out, a.seed, a.size)

    # Optional FID (graceful skip)
    real_images_dir = _find_real_class_dir(a.run_dir, a.cls)
    if real_images_dir:
        fid = _try_fid(real_images_dir, a.out, get_device())
        if fid is not None:
            print(f"[GAN] FID({a.cls}): {fid:.2f}")
            save_json(
                {"fid": fid, "cls": a.cls, "n": a.n},
                os.path.join(a.run_dir, f"fid_{a.cls}.json")
            )
        else:
            print(
                "[GAN] FID skipped -- install torch-fidelity or pytorch-fid to enable."
            )
    else:
        print("[GAN] FID skipped -- could not locate real train images directory.")


def _find_real_class_dir(run_dir, cls_name):
    """Try to find the real images directory for a class from split.csv paths."""
    manifest_path = os.path.join(run_dir, "split.csv")
    if not os.path.exists(manifest_path):
        return None
    splits = load_manifest(manifest_path)
    for p, l, _ in splits["train"]:
        if CLASSES[l] == cls_name:
            parent = os.path.dirname(p)
            if os.path.isdir(parent):
                return parent
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="GAN augmentation tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
subcommands:
  train   Train a GAN on a run's split.csv (train rows only)
  sample  Sample images from a trained generator
""",
    )
    ap.add_argument("subcommand", choices=["train", "sample"])
    args, rest = ap.parse_known_args(argv)
    if args.subcommand == "train":
        _cli_train(rest)
    else:
        _cli_sample(rest)


if __name__ == "__main__":
    main()
