import torch
import torch.nn as nn

class LCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(2)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
        nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.backbone(x)
        x = self.gap(x).flatten(1)
        return self.head(x)