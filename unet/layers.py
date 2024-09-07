import torch.nn as nn


class DepthwiseSeparableConv2D(nn.Module):
    """
    Performs Depthwise Separable 2D convolution.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, bias: bool = False
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding="same",
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
            bias=bias,
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DSConvBatchRelu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.dsconv = DepthwiseSeparableConv2D(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size
        )
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dsconv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x


class DownLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.net = nn.Sequential(
            DSConvBatchRelu(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
            ),
            # DSConvBatchRelu(
            #     in_channels=out_channels,
            #     out_channels=out_channels,
            #     kernel_size=kernel_size,
            # ),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class UpLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.net = nn.Sequential(
            # DSConvBatchRelu(
            #     in_channels=in_channels,
            #     out_channels=in_channels,
            #     kernel_size=kernel_size,
            # ),
            DSConvBatchRelu(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
            ),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )

    def forward(self, x):
        x = self.net(x)
        return x
