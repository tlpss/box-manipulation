import os
import unittest

import pytorch_lightning as pl
import torch

from keypoint_detection.src.datamodule import BoxKeypointsDataModule, BoxKeypointsDataset
from keypoint_detection.src.keypoint_utils import generate_keypoints_heatmap
from keypoint_detection.src.models import KeypointDetector


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
        batch_tensor = torch.Tensor([self.keypoints, self.keypoints])
        print(batch_tensor.shape)
        batch_heatmap = self.model.create_heatmap_batch((self.image_height, self.image_width), batch_tensor)
        self.assertEqual(batch_heatmap.shape, (2, self.image_height, self.image_width))


class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        self.model = KeypointDetector()

    def test_batch(self):
        """
        run train and evaluation to see if all goes as expected
        """
        model = KeypointDetector()
        TEST_DIR = os.path.dirname(os.path.abspath(__file__))

        module = BoxKeypointsDataModule(
            BoxKeypointsDataset(
                os.path.join(TEST_DIR, "test_dataset/dataset.json"), os.path.join(TEST_DIR, "test_dataset")
            ),
            2,
            0.5,  # make sure val dataloader has len >= 1
        )
        trainer = pl.Trainer(max_epochs=2, log_every_n_steps=1)
        trainer.fit(model, module)

        batch = next(iter(module.train_dataloader()))
        imgs, corner_keypoints, flap_keypoints = batch
        with torch.no_grad():
            model(imgs)

    # TODO: test on GPU if available
