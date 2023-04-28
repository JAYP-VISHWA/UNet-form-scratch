import torch
from torch import nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    """Double Cnvolution

    :param nn: pytorch nn module
    :type nn: module
    """

    def __init__(self, in_channels, out_channels):
        """Constructor

        :param in_channels: Number of channels in image (RGB)
        :type in_channels: int
        :param out_channels: Number of classes
        :type out_channels: int
        """
        # super(DoubleConv, self).__init__()
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Forward pass for Double Convolution

        :param x: data
        :type x: tensor
        :return: output of double convolution
        :rtype: tensor
        """
        return self.conv(x)


class UNET(nn.Module):
    """UNET class

    :param nn: nn module
    :type nn: torch module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        features=[64, 128, 256, 512],
    ):
        """Constructor for UNET class

        :param in_channels: Number of channels input
        :type in_channels: int
        :param out_channels: Number of output channels
        :type out_channels: int
        :param features: list of number of features in hidden channels,
        defaults to [64, 128, 256, 512]
        :type features: list, optional
        """
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downsampling
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2,
                    feature,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.finalconv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        """Forward pass for UNET

        :param x: Input data tensor
        :type x: tensor
        :return: output
        :rtype: tensor
        """
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.finalconv(x)


def test():
    """To test the UNET dimentions"""
    x = torch.randn((3, 1, 160, 160))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape


if __name__ == "__main__":
    test()
