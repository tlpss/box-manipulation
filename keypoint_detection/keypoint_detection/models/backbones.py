import abc
import argparse

import torch
import torch.nn as nn


class Backbone(nn.Module, abc.ABC):
    def __init__(self):
        super(Backbone, self).__init__()

    @abc.abstractmethod
    def get_n_channels_out(self):
        pass


class DilatedCnn(Backbone):
    def __init__(self, n_channels=32):
        super().__init__()
        self.n_channels_in = 3
        self.n_channels = n_channels
        kernel_size = (3, 3)
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=self.n_channels_in,
                out_channels=n_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                dilation=2,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                dilation=4,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                dilation=8,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                dilation=16,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                dilation=2,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                dilation=4,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                dilation=8,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                dilation=16,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.LeakyReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_n_channels_out(self):
        return self.n_channels


class ResNetBlock(nn.Module):
    """
    based on the basic ResNet Block used in torchvision
    inspired on https://jarvislabs.ai/blogs/resnet

    different from Peter in the position of the Norms?
    """

    def __init__(self, n_channels_in, n_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels_in, n_channels, kernel_size=(3, 3), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=(3, 3), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(n_channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class S3K(Backbone):
    """
    Backbone (approx) as in the S3K paper by Mel
    inspired by Peter's version of the backbone.
    """

    def __init__(self):
        self.kernel_size = (3, 3)
        super(S3K, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=self.kernel_size, padding="same")
        self.norm1 = nn.BatchNorm2d(num_features=3)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=self.kernel_size, stride=(2, 2))
        self.norm2 = nn.BatchNorm2d(num_features=16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=self.kernel_size, stride=(2, 2))
        self.norm3 = nn.BatchNorm2d(num_features=32)
        self.res1 = ResNetBlock(32)
        self.res2 = ResNetBlock(32)
        self.res3 = ResNetBlock(32)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel_size, padding="same")
        self.up1 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=self.kernel_size, stride=(2, 2))

        self.conv5 = nn.Conv2d(in_channels=32 + 16, out_channels=32, kernel_size=self.kernel_size, padding="same")
        self.norm4 = nn.BatchNorm2d(32)
        self.up2 = nn.ConvTranspose2d(
            in_channels=32, out_channels=32, kernel_size=self.kernel_size, stride=(2, 2), output_padding=1
        )
        self.conv6 = nn.Conv2d(in_channels=32 + 3, out_channels=32, kernel_size=self.kernel_size, padding="same")
        self.norm5 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x_0 = x
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x_1 = x
        x = self.norm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)

        x = self.res1(x)

        x = self.res2(x)

        x = self.res3(x)

        x = self.conv4(x)

        x = self.up1(x)
        x = torch.cat([x, x_1], dim=1)
        x = self.conv5(x)
        x = self.norm4(x)
        x = self.relu(x)

        x = self.up2(x)
        x = torch.cat([x, x_0], dim=1)
        x = self.conv6(x)
        x = self.norm5(x)
        x = self.relu(x)
        return x

    def get_n_channels_out(self):
        return 32


class BackboneFactory:
    @staticmethod
    def create_backbone(backbone: str, **kwargs) -> Backbone:
        if backbone == "DilatedCnn":
            return DilatedCnn(**kwargs)
        elif backbone == "S3K":
            return S3K()
        else:
            raise Exception("Unknown backbone type")

    @staticmethod
    def add_to_argparse(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("BackboneFactory")
        parser.add_argument("--backbone", type=str, default="DilatedCnn")
        return parent_parser
