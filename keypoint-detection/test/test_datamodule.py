import os
import unittest

from src.datamodule import BoxKeypointsDataModule, BoxKeypointsDataset


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

        # TODO: check format of batch components.
