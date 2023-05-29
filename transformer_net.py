from torch.nn import Module, Conv2d, ReflectionPad2d, InstanceNorm2d, ReLU
from torch.nn.functional import interpolate


class ConvLayer(Module):
    """
    Creating a custom convolutional layer with reflection padding before a conv2d layer
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2) -> None:
        super(ConvLayer, self).__init__()
        self.reflection_pad = ReflectionPad2d(stride//2)
        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        return self.conv(self.reflection_pad(x))


class ResidualBlock(Module):
    """Resnet style Residual Block
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: list[int] = [3, 3], stride: list[int] = [1, 1]) -> None:
        super(ResidualBlock, self).__init__()
        self.layers: list = []
        for i in range(2):
            self.layers.append(Conv2d(
                in_channels if i == 0 else out_channels, out_channels, kernel_size[i], stride[i]))
            self.layers.append(InstanceNorm2d(out_channels, affine=True))
        self.relu = ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.layers[1](self.layers[0](x)))
        out = self.layers[3](self.layers[2](x))
        out += x
        # return self.relu(out)
        return out


class UpsampleConv(Module):
    def __init__(self, in_channels, out_channels, kernel_size: int = 3, stride: int = 1, upsample: int = None) -> None:
        super(UpsampleConv, self).__init__()
        self.upsample = upsample
        self.reflection_pad = ReflectionPad2d(kernel_size//2)
        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = interpolate(x, mode='nearest', scale_factor=self.upsample)
        out = self.conv(self.reflection_pad(x))
        return out


class TransformerNet(Module):
    def __init__(self) -> None:
        super(TransformerNet, self).__init__()
        # Convolutional layers
        self.conv_1 = ConvLayer(3, 32, 9, 1)
        self.instance_norm_1 = InstanceNorm2d(32, affine=True)
        self.conv_2 = ConvLayer(32, 64)
        self.instance_norm_2 = InstanceNorm2d(64, affine=True)
        self.conv_3 = ConvLayer(64, 128)
        self.instance_norm_3 = InstanceNorm2d(128, affine=True)

        # Residual layers
        self.res_1 = ResidualBlock(128, 128)
        self.res_2 = ResidualBlock(128, 128)
        self.res_3 = ResidualBlock(128, 128)
        self.res_4 = ResidualBlock(128, 128)
        self.res_5 = ResidualBlock(128, 128)

        # Upsample layers
        self.upsample_1 = UpsampleConv(128, 64, upsample=2)
        self.instance_norm_4 = InstanceNorm2d(64, affine=True)
        self.upsample_2 = UpsampleConv(64, 32)
        self.instance_norm_5 = InstanceNorm2d(32, affine=True)
        self.upsample_3 = UpsampleConv(32, 3, 9)

        self.relu = ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.instance_norm_1(self.conv_1(x)))
        out = self.relu(self.instance_norm_2(self.conv_2(out)))
        out = self.relu(self.instance_norm_3(self.conv_3(out)))

        out = self.res_1(out)
        out = self.res_2(out)
        out = self.res_3(out)
        out = self.res_4(out)
        out = self.res_5(out)

        out = self.relu(self.instance_norm_4(self.upsample_1))
        out = self.relu(self.instance_norm_5(self.upsample_2))
        out = self.upsample_3(out)

        return out
