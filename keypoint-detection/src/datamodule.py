import json
import os

import pytorch_lightning as pl
import torch
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
        Note: only works for squared Images!
        """
        keypoints *= image_shape
        keypoints[:, -1] = image_shape - keypoints[:, -1]
        return keypoints

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        index = int(index)

        image_path = os.path.join(os.getcwd(), self.image_dir, self.dataset[index]["image_path"])
        image = io.imread(image_path)
        image = self.transform(image)

        corner_keypoints = torch.Tensor(self.dataset[index]["corner_keypoints"])
        if self.flap_keypoints_type == "center":
            flap_keypoints = torch.Tensor(self.dataset[index]["flap_center_keypoints"])
        else:
            flap_keypoints = torch.Tensor(self.dataset[index]["flap_corner_keypoints"])

        # convert keypoints to pixel coords

        corner_keypoints = self.convert_keypoint_coordinates_to_pixel_coordinates(corner_keypoints, image.shape[-1])
        flap_keypoints = self.convert_keypoint_coordinates_to_pixel_coordinates(flap_keypoints, image.shape[-1])
        return image, corner_keypoints, flap_keypoints


class BoxKeypointsDataModule(pl.LightningDataModule):
    def __init__(self, dataset: BoxKeypointsDataset, batch_size: int = 4):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

        # TODO: split in test and train set
        self.train_dataset = dataset

    def train_dataloader(self):
        dataloader = DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=4)
        return dataloader

    def test_dataloader(self):
        pass