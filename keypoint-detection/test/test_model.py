import unittest

import pytorch_lightning as pl
import torch
from src.datamodule import BoxKeypointsDataModule, BoxKeypointsDataset
from src.keypoint_utils import generate_keypoints_heatmap
from src.models import KeypointDetector


class TestHeatmapUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.image_width = 32
        self.image_height = 16
        self.keypoints = [[10, 4], [10, 8], [30, 7]]
        self.sigma = 3

        self.heatmaps = generate_keypoints_heatmap((self.image_height, self.image_width), self.keypoints, self.sigma)
        self.model = KeypointDetector()

    def test_perfect_heatmap(self):
        loss = self.model.heatmap_loss(self.heatmaps, self.heatmaps)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss >= 0)

    def test_heatmap_batch(self):
        # TODO
        pass


class TestModelOverfit(unittest.TestCase):
    def setUp(self) -> None:
        self.model = KeypointDetector()

    def test_overfit(self):
        model = KeypointDetector()
        # todo: fix paths
        module = BoxKeypointsDataModule(BoxKeypointsDataset("./mock_dataset/dataset.json", "./mock_dataset/images"), 1)
        trainer = pl.Trainer(max_epochs=20)
        trainer.fit(model, module)

        self.assertTrue(model(next(iter(module))))
