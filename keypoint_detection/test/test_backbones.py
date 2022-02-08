import unittest

import torch

from keypoint_detection.models.backbones.dilated_cnn import DilatedCnn
from keypoint_detection.models.backbones.s3k import S3K
from keypoint_detection.models.backbones.unet import UnetBackbone


class TestBackbones(unittest.TestCase):
    def setUp(self) -> None:
        self.x = torch.randn((4, 3, 64, 64))

    def test_s3k(self):
        backbone = S3K()

        output = backbone(self.x)
        self.assertEqual((output.shape), (4, 32, 64, 64))

    def test_dilated_cnn(self):
        backbone = DilatedCnn()
        output = backbone(self.x)
        self.assertEqual((output.shape), (4, 32, 64, 64))

    def test_unet(self):
        backbone = UnetBackbone(
            n_channels_in=3, n_downsampling_layers=2, n_resnet_blocks=2, channels=4, kernel_size=3, dilation=1, test=3
        )
        output = backbone(self.x)
        self.assertEqual((output.shape), (4, 4, 64, 64))
