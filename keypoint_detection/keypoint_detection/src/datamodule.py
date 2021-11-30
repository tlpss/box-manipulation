import json
import os
import random
import time

import pytorch_lightning as pl
import torch
import tqdm
from skimage import io
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor


class BoxKeypointsDataset(Dataset):
    """
    Create Custom Pytorch Dataset from the Box dataset
    cf https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(self, json_file: str, image_dir: str, flap_keypoints_type: str = "corner"):
        """
        json_file:  path to json file with dataset
        image_dir: path to dir from where the relative image paths in the json are included
        flap_keypoints: 'corner' or 'center', determines which type of keypoints that will be used for the flaps
        """

        self.json_file = json_file
        self.image_dir = image_dir

        assert flap_keypoints_type in ["center", "corner"]
        self.flap_keypoints_type = flap_keypoints_type

        self.transform = ToTensor()  # convert images to Torch Tensors

        f = open(json_file, "r")

        # dataset format is defined in project README
        self.dataset = json.load(f)
        self.dataset = self.dataset["dataset"]

    def __len__(self):
        return len(self.dataset)

    def convert_keypoint_coordinates_to_pixel_coordinates(
        self, keypoints: torch.Tensor, image_shape: int
    ) -> torch.Tensor:
        """
        Converts the keypoint coordinates as generated in Blender to (u,v) coordinates with the origin in the top left corner and the u-axis going right.
        Note: only works for squared Images!
        """
        keypoints *= image_shape
        keypoints[:, -1] = image_shape - keypoints[:, -1]
        return keypoints

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        index = int(index)

        # load images @runtime
        image_path = os.path.join(os.getcwd(), self.image_dir, self.dataset[index]["image_path"])
        image = io.imread(image_path)
        image = self.transform(image)

        # read keypoints
        corner_keypoints = torch.Tensor(self.dataset[index]["corner_keypoints"])
        if self.flap_keypoints_type == "center":
            flap_keypoints = torch.Tensor(self.dataset[index]["flap_center_keypoints"])
        else:
            flap_keypoints = torch.Tensor(self.dataset[index]["flap_corner_keypoints"])

        # convert keypoints to pixel coords
        corner_keypoints = self.convert_keypoint_coordinates_to_pixel_coordinates(corner_keypoints, image.shape[-1])
        flap_keypoints = self.convert_keypoint_coordinates_to_pixel_coordinates(flap_keypoints, image.shape[-1])

        return image, corner_keypoints, flap_keypoints


class DatasetIOCatcher(Dataset):
    """
    This Decorator for a Pytorch Dataset performs n attempts to load the dataset item, in an attempt
    to overcome IOErrors on the GPULab. This does not require the entire dataset to be in memory.
    """

    def __init__(self, dataset: Dataset, n_io_attempts: int):
        self.dataset = dataset
        self.n_io_attempts = n_io_attempts

    def __getitem__(self, index):
        sleep_time_in_seconds = 1
        for j in range(self.n_io_attempts):
            try:
                item = self.dataset[index]
                return item
            except IOError:
                if j == self.n_io_attempts - 1:
                    raise IOError(f"Could not preload item with index {index}")

                sleep_time = max(random.gauss(sleep_time_in_seconds, j), 0)
                print(f"caught IOError in {j}th attempt for item {index}, sleeping for {sleep_time} seconds")
                time.sleep(sleep_time)
                sleep_time_in_seconds *= 2

    def __len__(self):
        return len(self.dataset)


class DatasetPreloader(Dataset):
    """
    Decorator Pattern for a Pytorch Dataset where the dataset is preloaded into memory.
    This requires the whole dataset to fit into memory..

    There are 2 reasons for using this class:
    1. acces becomes faster during training
    2. GPULab throws IOErrors every now and then..
    """

    def __init__(self, dataset: Dataset, n_io_attempts: int):
        self.dataset = dataset
        self.preloaded_dataset = [None] * len(dataset)
        self._preload(n_io_attempts)

    def __getitem__(self, index):
        return self.preloaded_dataset[index]

    def __len__(self):
        return len(self.dataset)

    def _preload(self, io_attempts: int):
        """
        Attempt to load entire dataset into memory
        """
        for i in tqdm.trange(len(self)):
            for j in range(io_attempts):
                try:
                    self.preloaded_dataset[i] = self.dataset[i]
                    break
                except IOError:
                    print(f"caught IOError in {j}th attempt for item {i}")

                if j == io_attempts - 1:
                    raise IOError(f"Could not preload item with index {i}")


class BoxKeypointsDataModule(pl.LightningDataModule):
    def __init__(self, dataset: BoxKeypointsDataset, batch_size: int = 4, validation_split_ratio=0.1):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

        validation_size = int(validation_split_ratio * len(self.dataset))
        train_size = len(self.dataset) - validation_size
        self.train_dataset, self.validation_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, validation_size]
        )

    def train_dataloader(self):
        dataloader = DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=1)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(self.validation_dataset, self.batch_size, shuffle=True, num_workers=1)
        return dataloader

    def test_dataloader(self):
        pass
