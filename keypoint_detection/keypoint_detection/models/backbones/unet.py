import torch
import torch.nn as nn
import math 
import argparse
from keypoint_detection.models.backbones.base_backbone import Backbone
from keypoint_detection.models.backbones.s3k import ResNetBlock

class StridedDownSamplingBlock(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, kernel_size, dilation):
        super().__init__()
        padding = math.floor((dilation * (kernel_size -1) +1)/2 -1) +1 
        self.conv = nn.Conv2d(in_channels=n_channels_in, out_channels=n_channels_out, kernel_size=kernel_size,stride=2, dilation=dilation, padding=padding, bias=False)
        self.norm = nn.BatchNorm2d(n_channels_out)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class UpSamplingBlock(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, kernel_size):
        super().__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Conv2d(in_channels=n_channels_in*2, out_channels=n_channels_out, kernel_size=kernel_size,bias=False, padding = 'same')
        self.norm = nn.BatchNorm2d(n_channels_out)
        self.relu = nn.ReLU()
    def forward(self,x,x_skip):
        x = self.upsample(x)
        x = torch.cat([x,x_skip],dim=1)
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class UnetBackbone(Backbone):
    def __init__(self, n_channels_in,n_downsampling_layers, n_resnet_blocks, channels, kernel_size, dilation, **kwargs):
        super().__init__()
        self.n_channels = channels
        self.conv1 = nn.Conv2d(n_channels_in, channels, kernel_size, padding="same")
        self.downsampling_blocks = [StridedDownSamplingBlock(channels,channels,kernel_size,dilation) for _ in range(n_downsampling_layers)]
        self.resnet_blocks = [ResNetBlock(channels,channels) for _ in range(n_resnet_blocks)]
        self.upsampling_blocks = [UpSamplingBlock(n_channels_in=channels,n_channels_out=channels,kernel_size=kernel_size) for _ in range(n_downsampling_layers)]

    def forward(self,x):
        skips = []

        x = self.conv1(x)

        for block in self.downsampling_blocks:
            skips.append(x)
            x = block(x)
            print(x.shape)

        for block in self.resnet_blocks:
            x = block(x)
        
        for block in self.upsampling_blocks:
            x_skip = skips.pop()
            x = block(x,x_skip)
            print(x.shape)
        return x

    def get_n_channels_out(self):
        return self.n_channels

    @staticmethod
    def add_to_argparse(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("UnetBackbone")
        parser.add_argument("--n_channels_in", type=int, default=3)
        parser.add_argument("--n_channels", type=int, default =16)
        parser.add_argument("--n_resnet_blocks", type=int, default = 3)
        parser.add_argument("--n_downsampling_layers", type=int, default = 2)
        parser.add_argument("--kernel_size", type=int, default =3)
        parser.add_argument("--dilation", type=int, default = 1)

        return parent_parser


if __name__ == '__main__':
    x = torch.rand(2,3,64,64)
    model = UnetBackbone(3,3,3,16,3,2)
    y = model(x)
