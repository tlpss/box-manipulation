import argparse

import torch
import torch.nn as nn


class DilatedCnn(nn.Module):
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


class BackboneFactory:
    @staticmethod
    def create_backbone(backbone: str, **kwargs) -> nn.Module:
        if backbone == "DilatedCnn":
            return DilatedCnn(**kwargs)
        else:
            raise Exception("Unknown backbone type")

    @staticmethod
    def add_to_argparse(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("BackboneFactory")
        parser.add_argument("--backbone", type=str, default="DilatedCnn")
