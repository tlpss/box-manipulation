import unittest

import torch

from keypoint_detection.models.backbones.dilated_cnn import DilatedCnn
from keypoint_detection.models.backbones.s3k import S3K


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
