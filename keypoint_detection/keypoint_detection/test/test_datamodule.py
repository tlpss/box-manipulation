import os
import unittest

import numpy as np
import torch

from keypoint_detection.src.datamodule import BoxDatasetPreloaded, BoxKeypointsDataModule, BoxKeypointsDataset


class TestDataSet(unittest.TestCase):
    def setUp(self):
        TEST_DIR = os.path.dirname(os.path.abspath(__file__))
        self.json_path = os.path.join(TEST_DIR, "test_dataset/dataset.json")
        self.image_path = os.path.join(TEST_DIR, "test_dataset")

    def test_dataset(self):

        dataset = BoxKeypointsDataset(self.json_path, self.image_path)
        item = dataset.__getitem__(0)
        img, corner, flap = item
        self.assertEqual(img.shape, (3, 256, 256))
        self.assertEqual(len(corner), 4)
        self.assertEqual(len(flap), 8)

        dataset = BoxKeypointsDataset(self.json_path, self.image_path, flap_keypoints_type="center")
        item = dataset.__getitem__(0)
        img, corner, flap = item
        self.assertEqual(len(flap), 4)


class TestDatasetPreloader(unittest.TestCase):
    def setUp(self):
        TEST_DIR = os.path.dirname(os.path.abspath(__file__))
        self.json_path = os.path.join(TEST_DIR, "test_dataset/dataset.json")
        self.image_path = os.path.join(TEST_DIR, "test_dataset")
        self.dataset = BoxKeypointsDataset(self.json_path, self.image_path)

    def test_preloader(self):
        preloadeded_dataset = BoxDatasetPreloaded(self.json_path, self.image_path, n_io_attempts=2)
        self.assertEqual(len(preloadeded_dataset.__getitem__(1)), 3)
        self.assertIsNotNone(preloadeded_dataset.preloaded_images[0])
        self.assertTrue(isinstance(preloadeded_dataset.preloaded_images[0], np.ndarray))


class TestDataModule(unittest.TestCase):
    def setUp(self):
        TEST_DIR = os.path.dirname(os.path.abspath(__file__))
        self.json_path = os.path.join(TEST_DIR, "test_dataset/dataset.json")
        self.image_path = os.path.join(TEST_DIR, "test_dataset")
        self.dataset = BoxKeypointsDataset(self.json_path, self.image_path)

    def test_split(self):
        module = BoxKeypointsDataModule(self.dataset, batch_size=1)
        train_dataloader = module.train_dataloader()
        self.assertEqual(len(train_dataloader), 2)

        module = BoxKeypointsDataModule(self.dataset, batch_size=1, validation_split_ratio=0.5)
        train_dataloader = module.train_dataloader()
        validation_dataloader = module.train_dataloader()
        self.assertEqual(len(train_dataloader), 1)
        self.assertEqual(len(validation_dataloader), 1)

    def test_batch_format(self):
        module = BoxKeypointsDataModule(self.dataset, batch_size=1)
        train_dataloader = module.train_dataloader()

        batch = next(iter(train_dataloader))
        self.assertEqual(len(batch), 3)

        img, corner_kp, flap_kp = batch

        self.assertIsInstance(img, torch.Tensor)
        self.assertEquals(img.shape, (1, 3, 256, 256))
        self.assertIsInstance(corner_kp, torch.Tensor)
        self.assertEquals(corner_kp.shape, (1, 4, 2))
        self.assertIsInstance(flap_kp, torch.Tensor)
        self.assertEquals(flap_kp.shape, (1, 8, 2))
