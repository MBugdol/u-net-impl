from .layers import DownLayer, UpLayer
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, kernel_size: int, num_classes: int):
        super().__init__()

        self.num_classes = num_classes

        self.down1 = DownLayer(3, 64, kernel_size=kernel_size)
        self.down2 = DownLayer(64, 128, kernel_size=kernel_size)
        self.down3 = DownLayer(128, 256, kernel_size=kernel_size)

        self.up3 = UpLayer(256, 128, kernel_size=kernel_size)
        self.up2 = UpLayer(128, 64, kernel_size=kernel_size)
        self.up1 = UpLayer(64, num_classes, kernel_size=kernel_size)

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)

        return x
