import os
import unittest

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

from keypoint_detection.data.datamodule import BoxKeypointsDataModule
from keypoint_detection.data.dataset import BoxKeypointsDataset
from keypoint_detection.models.metrics import KeypointAPMetric
from keypoint_detection.models.models import KeypointDetector
from keypoint_detection.utils.heatmap import generate_keypoints_heatmap


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
        wandb_logger = WandbLogger(dir=KeypointDetector.get_wand_log_dir_path(), mode="offline")

        model = KeypointDetector(maximal_gt_keypoint_pixel_distances=[2, 4, 5])
        TEST_DIR = os.path.dirname(os.path.abspath(__file__))

        module = BoxKeypointsDataModule(
            BoxKeypointsDataset(
                os.path.join(TEST_DIR, "test_dataset/dataset.json"), os.path.join(TEST_DIR, "test_dataset")
            ),
            2,
            0.5,  # make sure val dataloader has len >= 1
        )
        if torch.cuda.is_available():
            gpus = 1
        else:
            gpus = 0

        trainer = pl.Trainer(max_epochs=2, log_every_n_steps=1, gpus=gpus, logger=wandb_logger)
        trainer.fit(model, module)

        batch = next(iter(module.train_dataloader()))
        imgs, corner_keypoints, flap_keypoints = batch
        with torch.no_grad():
            model(imgs)

    def test_gt_heatmaps(self):
        max_dst = 2
        model = KeypointDetector(heatmap_sigma=8)
        metric = KeypointAPMetric(max_dst)
        TEST_DIR = os.path.dirname(os.path.abspath(__file__))
        module = BoxKeypointsDataModule(
            BoxKeypointsDataset(
                os.path.join(TEST_DIR, "test_dataset/dataset.json"), os.path.join(TEST_DIR, "test_dataset")
            ),
            2,
            0.5,  # make sure val dataloader has len >= 1
        )
        for batch in module.train_dataloader():
            imgs, corner_keypoints, _ = batch
            heatmaps = model.create_heatmap_batch(imgs[0].shape[1:], corner_keypoints)
            model.update_ap_metrics(heatmaps, corner_keypoints, metric)

        ap = metric.compute()
        self.assertEqual(ap, 1.0)
