import torch.nn as nn
from torchvision import models

ARCHS = ("smallcnn", "resnet18", "resnet50")


def small_cnn():
    def blk(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                             nn.ReLU(), nn.MaxPool2d(2))
    return nn.Sequential(blk(3, 16), blk(16, 32), blk(32, 64), blk(64, 128),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Dropout(0.3), nn.Linear(128, 2))


def build(name, pretrained=True):
    if name == "smallcnn":
        return small_cnn()
    if name == "resnet18":
        m = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    elif name == "resnet50":
        m = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
    else:
        raise ValueError(f"unknown arch {name}; choose from {ARCHS}")
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m


def target_layer(model):
    """Last convolutional stage, used by Grad-CAM."""
    return model.layer4 if hasattr(model, "layer4") else model[3]
