import abc
import argparse
import distutils.util
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import tqdm
from skimage import io
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from keypoint_detection.utils.tensor_padding import pad_tensor_with_nans


class ImageDataset(Dataset, abc.ABC):
    def __init__(self):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    def get_image(self, index: int) -> np.ndarray:
        """
        get image associated to dataset[index]
        """


class BoxKeypointsDataset(ImageDataset):
    """
    Create Custom Pytorch Dataset from the Box dataset
    cf https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    @staticmethod
    def add_argparse_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        add named arguments from the init function to the parser
        The default values here are actually duplicates from the init function, but this was for readability (??)
        """
        parser = parent_parser.add_argument_group("BoxkeypointsDataset")
        parser.add_argument("--image_dataset_path", required=False, type=str)
        parser.add_argument("--json_dataset_path", required=False, type=str)
        parser.add_argument("--flap_keypoints_type", required=False, type=str)
        parser.add_argument("--non_occluded_keypoints_only", type=distutils.util.strtobool)

        return parent_parser

    def __init__(
        self,
        json_dataset_path: str,
        image_dataset_path: str,
        flap_keypoints_type: str = "corner",
        non_occluded_keypoints_only: bool = False,
        **kwargs,
    ):
        """
        json_file:  path to json file with dataset
        image_dir: path to dir from where the relative image paths in the json are included
        flap_keypoints: 'corner' or 'center', determines which type of keypoints that will be used for the flaps
        """
        super(BoxKeypointsDataset, self).__init__()
        self.json_file = json_dataset_path
        self.image_dir = image_dataset_path

        assert flap_keypoints_type in ["center", "corner"]
        self.flap_keypoints_type = flap_keypoints_type

        self.corner_keypoints_name = "corner_keypoints"
        self.flap_keypoints_name = f"flap_{flap_keypoints_type}_keypoints"

        self.non_occluded_keypoints_only = non_occluded_keypoints_only
        if non_occluded_keypoints_only:
            self.corner_keypoints_name += "_visible"
            self.flap_keypoints_name += "_visible"

        self.transform = ToTensor()  # convert images to Torch Tensors

        f = open(json_dataset_path, "r")

        # dataset format is defined in project README
        self.dataset = json.load(f)
        self.dataset = self.dataset["dataset"]

    def __len__(self):
        return len(self.dataset)

    @classmethod
    def get_data_dir_path(cls) -> Path:
        return Path(__file__).resolve().parents[2] / "datasets"

    def convert_keypoint_coordinates_to_pixel_coordinates(
        self, keypoints: torch.Tensor, image_shape: int
    ) -> torch.Tensor:
        """
        Converts the keypoint coordinates as generated in Blender to (u,v) coordinates
        with the origin in the top left corner and the u-axis going right.
        Note: only works for squared Images!
        """
        keypoints *= image_shape
        keypoints[:, 1] = image_shape - keypoints[:, 1]
        return keypoints

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        index = int(index)

        image = self.get_image(index)
        image = self.transform(image)

        # read keypoints
        corner_keypoints = torch.Tensor(self.dataset[index][self.corner_keypoints_name])
        flap_keypoints = torch.Tensor(self.dataset[index][self.flap_keypoints_name])

        # convert keypoints to pixel coords
        corner_keypoints = self.convert_keypoint_coordinates_to_pixel_coordinates(corner_keypoints, image.shape[-1])
        flap_keypoints = self.convert_keypoint_coordinates_to_pixel_coordinates(flap_keypoints, image.shape[-1])

        # pad the tensor to make sure all items of a batch have same size and can hence be collated by the default
        # torch collate function.
        if self.non_occluded_keypoints_only:
            corner_keypoints = pad_tensor_with_nans(corner_keypoints, 4)
            flap_keypoints = pad_tensor_with_nans(flap_keypoints, 8 if self.flap_keypoints_type == "corner" else 4)

        return image, corner_keypoints, flap_keypoints

    def get_image(self, index: int) -> np.ndarray:
        """
        read the image from disk and return as np array
        """
        # load images @runtime from disk
        image_path = os.path.join(os.getcwd(), self.image_dir, self.dataset[index]["image_path"])
        image = io.imread(image_path)
        return image


class BoxDatasetIOCatcher(BoxKeypointsDataset):
    """
    This Dataset performs n attempts to load the dataset item, in an attempt
    to overcome IOErrors on the GPULab. This does not require the entire dataset to be in memory.
    """

    @staticmethod
    def add_argparse_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return BoxKeypointsDataset.add_argparse_args(parent_parser)

    def __init__(
        self,
        json_dataset_path: str,
        image_dataset_path: str,
        flap_keypoints_type: str = "corner",
        non_occluded_keypoints_only: bool = False,
        n_io_attempts: int = 4,
        **kwargs,
    ):
        """
        n_io_attempts: number of trials to load image from IO
        """
        super().__init__(
            json_dataset_path, image_dataset_path, flap_keypoints_type, non_occluded_keypoints_only, **kwargs
        )
        self.n_io_attempts = n_io_attempts

    def get_image(self, index: int) -> np.ndarray:
        sleep_time_in_seconds = 1
        for j in range(self.n_io_attempts):
            try:
                image = super().get_image(index)  # IO read.
                return image
            except IOError:
                if j == self.n_io_attempts - 1:
                    raise IOError(f"Could not load image for dataset entry with index {index}")

                sleep_time = max(random.gauss(sleep_time_in_seconds, j), 0)
                print(
                    f"caught IOError in {j}th attempt to load image for item {index}, sleeping for {sleep_time} seconds"
                )
                time.sleep(sleep_time)
                sleep_time_in_seconds *= 2


class BoxDatasetPreloaded(BoxDatasetIOCatcher):
    """
    The images are preloaded in memory for faster access.
    This requires the whole dataset to fit into memory, so make sure to have enough memory available.

    """

    @staticmethod
    def add_argparse_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return BoxDatasetIOCatcher.add_argparse_args(parent_parser)

    def __init__(
        self,
        json_dataset_path: str,
        image_dataset_path: str,
        flap_keypoints_type: str = "corner",
        non_occluded_keypoints_only: bool = False,
        n_io_attempts: int = 4,
        **kwargs,
    ):
        """
        n_io_attempts: number of trials to load image from IO
        """
        super().__init__(
            json_dataset_path,
            image_dataset_path,
            flap_keypoints_type,
            non_occluded_keypoints_only,
            n_io_attempts,
            **kwargs,
        )
        self.preloaded_images = [None] * len(self.dataset)
        self._preload()

    def _preload(self):
        """
        load images into memory as np.ndarrays.
        Choice to load them as np.ndarrays is because pytorch uses float32 for each value whereas
        the original values are only 8 bit ints, so this is a 4times increase in size..
        """

        print("preloading dataset images")
        for i in tqdm.trange(len(self)):
            self.preloaded_images[i] = super().get_image(i)
        print("dataset images preloaded")

    def get_image(self, index: int) -> np.ndarray:
        return self.preloaded_images[index]
