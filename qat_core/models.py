import torch
import torch.nn as nn
from .layers import QATConv2d

class QATNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.block1 = nn.Sequential(
            QATConv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        self.block2 = nn.Sequential(
            QATConv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        output = self.fc(x)
        return output