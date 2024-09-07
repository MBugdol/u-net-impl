"""Contains definitions used for the Camouflaged Animals Model"""

from torch import nn, Tensor


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        # Performs a spatial (depthwise) convolution on each channel seperately
        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            bias=bias,
            groups=in_channels,
        )
        # Performs a non-spatial (pointwise) joining convolution on the tensor
        self.pointwise_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=bias
        )

        nn.init.xavier_uniform_(self.depthwise_conv.weight)
        nn.init.xavier_uniform_(self.pointwise_conv.weight)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class DSConvBatchRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        self.sequence = nn.Sequential(
            DepthwiseSeparableConv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.sequence(x)


class DownDSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, iterations):
        super().__init__()
        # first iteration changes channel count
        self.initial_seq = DSConvBatchRelu(
            in_channels, out_channels, kernel_size=kernel_size, bias=False
        )
        # following iterations-1 iterations keep channel count the same
        self.iterative_seq = nn.Sequential(
            *[
                DSConvBatchRelu(
                    out_channels, out_channels, kernel_size=kernel_size, bias=False
                )
                for _ in range(iterations - 1)
            ]
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, return_indices=True)

    def forward(self, x):
        y = self.initial_seq(x)
        y = self.iterative_seq(y)

        shape = y.shape
        y, indices = self.maxpool(y)
        return y, indices, shape


class UpDSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, iterations):
        super().__init__()
        # first iterations-1 iterations keep channel count the same
        self.iterative_seq = nn.Sequential(
            *[
                DSConvBatchRelu(
                    in_channels, in_channels, kernel_size=kernel_size, bias=False
                )
                for _ in range(iterations - 1)
            ]
        )
        # last iteration changes channel count
        self.last_seq = DSConvBatchRelu(
            in_channels, out_channels, kernel_size=kernel_size, bias=False
        )
        self.maxunpool = nn.MaxUnpool2d(kernel_size=2)

    def forward(self, x, indices, output_size):
        y = self.maxunpool(x, indices, output_size=output_size)
        y = self.iterative_seq(y)
        y = self.last_seq(y)
        return y


class UNet(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.bn_input = nn.BatchNorm2d(3)

        self.dconv1 = DownDSConv(3, 64, kernel_size=kernel_size, iterations=3) # 256 - 128
        self.dconv2 = DownDSConv(64, 128, kernel_size=kernel_size, iterations=3) #128 - 64
        self.dconv3 = DownDSConv(128, 256, kernel_size=kernel_size, iterations=4) # 64 - 32
        # self.dconv4 = DownDSConv(256, 512, kernel_size=kernel_size, iterations=3) # 32 - 16
        
        # self.uconv4 = UpDSConv(512, 256, kernel_size=kernel_size, iterations=3)
        self.uconv3 = UpDSConv(256, 128, kernel_size=kernel_size, iterations=4)
        self.uconv2 = UpDSConv(128, 64, kernel_size=kernel_size, iterations=3)
        self.uconv1 = UpDSConv(64, 4, kernel_size=kernel_size, iterations=3)

    def forward(self, batch: Tensor):
        x = self.bn_input(batch)

        unpool_stack = []
        for down_layer in [self.dconv1, self.dconv2, self.dconv3]:
            x, *args = down_layer(x)
            unpool_stack.append(args)
        
        for up_layer in [self.uconv3, self.uconv2, self.uconv1]:
            x = up_layer(x, *unpool_stack.pop())
            
        x = nn.Softmax(dim=1)(x)
        # x = nn.SoftMax

        return x