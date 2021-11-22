from typing import List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from src.keypoint_utils import generate_keypoints_heatmap, get_keypoints_from_heatmap
from src.metrics import DetectedKeypoint, Keypoint, KeypointmAPMetric


class KeypointDetector(pl.LightningModule):
    """
    Box keypoint Detector using Gaussian Heatmaps
    There are 2 channels (groups) of keypoints, one that detects box corners and
    one that detects the center (or corners) of the outer edges of all flaps of the box.

    """

    def __init__(
        self,
        heatmap_sigma=10,
        n_channels=32,
        detect_flap_keypoints=True,
        minimal_keypoint_extraction_pixel_distance: int = None,
        maximal_gt_keypoint_pixel_distance: int = None,
    ):
        """[summary]

        Args:
            heatmap_sigma (int, optional): Sigma of the gaussian heatmaps used to train the detector. Defaults to 10.
            n_channels (int, optional): Number of channels for the CNN layers. Defaults to 32.
            detect_flap_keypoints (bool, optional): Detect flap keypoints in a second channel or use a single channel Detector for box corners only.
            minimal_keypoint_extraction_pixel_distance (int, optional): the minimal distance (in pixels) between two detected keypoints,
                                                                        or the size of the local mask in which a keypoint needs to be the local maximum
            maximal_gt_keypoint_pixel_distance (int, optional): the maximal distance between a gt keypoint and detected keypoint, for the keypoint to be considered a TP
        """
        super().__init__()
        ## No need to manage devices ourselves, pytorch.lightning does all of that.
        ## device can be accessed through self.device if required.
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.detect_flap_keypoints = detect_flap_keypoints
        self.heatmap_sigma = heatmap_sigma

        if minimal_keypoint_extraction_pixel_distance:
            self.minimal_keypoint_pixel_distance = minimal_keypoint_extraction_pixel_distance
        else:
            self.minimal_keypoint_pixel_distance = heatmap_sigma

        if maximal_gt_keypoint_pixel_distance:
            self.maximal_gt_keypoint_pixel_distance = maximal_gt_keypoint_pixel_distance
        else:
            self.maximal_gt_keypoint_pixel_distance = heatmap_sigma
        self.validation_metric = KeypointmAPMetric(self.maximal_gt_keypoint_pixel_distance)

        self.n_channels_in = 3
        self.n_channes = n_channels
        self.n_channels_out = (
            2 if self.detect_flap_keypoints else 1
        )  # number of keypoint classes = number of output channels of CNN
        kernel_size = (3, 3)
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=self.n_channels_in,
                out_channels=n_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                dilation=2,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                dilation=4,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                dilation=8,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                dilation=16,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                dilation=2,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                dilation=4,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                dilation=8,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                dilation=16,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=self.n_channels_out,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.Sigmoid(),
        ).to(self.device)

    def forward(self, x: torch.Tensor):
        """
        x shape must be (N,C_in,H,W) with N batch size, and C_in number of incoming channels (3)
        return shape = (N, 1, H,W)
        """
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def heatmap_loss(self, predicted_heatmaps: torch.Tensor, heatmaps: torch.Tensor) -> torch.Tensor:
        """Computes the loss of 2 batches of heatmaps

        Args:
            predicted_heatmaps (torch.Tensor(NxHxW)):the predicted heatmaps
            heatmaps (torch.Tensor((NxHxW)): the ground truth heatmaps

        Returns:
            torch.Tensor: scalar loss value
        """
        # No focal loss (Objects as Points) as in CenterNet paper but BCS as in PAF
        # bc @Peter said it does not improve performance too much (KISS)
        return torch.nn.functional.binary_cross_entropy(predicted_heatmaps, heatmaps, reduction="mean")

    def training_step(self, train_batch, batch_idx):
        imgs, corner_keypoints, flap_keypoints = train_batch

        # load here to device to keep mem consumption low, if possible one could also load entire dataset on GPU to speed up training..
        imgs = imgs.to(self.device)

        ## predict and compute losses
        corner_heatmaps = self.create_heatmap_batch(imgs[0].shape[1:], corner_keypoints)
        predicted_heatmaps = self.forward(imgs)  # create heatmaps JIT, is this desirable?
        predicted_corner_heatmaps = predicted_heatmaps[:, 0, :, :]
        corner_loss = self.heatmap_loss(predicted_corner_heatmaps, corner_heatmaps)
        loss = corner_loss

        if self.detect_flap_keypoints:
            flap_heatmaps = self.create_heatmap_batch(imgs[0].shape[1:], flap_keypoints)
            predicted_flap_heatmaps = predicted_heatmaps[:, 1, :, :]
            flap_loss = self.heatmap_loss(predicted_flap_heatmaps, flap_heatmaps)
            loss += flap_loss

        ## logging
        self.log("train_corner_loss", corner_loss)
        self.log("train_loss", loss)
        if self.detect_flap_keypoints:
            self.log("train_flap_loss", flap_loss)

        return loss

    def validation_step(self, val_batch, batch_idx):

        imgs, corner_keypoints, flap_keypoints = val_batch

        # load here to device to keep mem consumption low, if possible one could also load entire dataset on GPU to speed up training..
        imgs = imgs.to(self.device)

        with torch.no_grad():
            ## predict and compute losses
            corner_heatmaps = self.create_heatmap_batch(imgs[0].shape[1:], corner_keypoints)
            predicted_heatmaps = self.forward(imgs)  # create heatmaps JIT, is this desirable?
            predicted_corner_heatmaps = predicted_heatmaps[:, 0, :, :]
            corner_loss = self.heatmap_loss(predicted_corner_heatmaps, corner_heatmaps)
            loss = corner_loss

            if self.detect_flap_keypoints:
                flap_heatmaps = self.create_heatmap_batch(imgs[0].shape[1:], flap_keypoints)
                predicted_flap_heatmaps = predicted_heatmaps[:, 1, :, :]
                flap_loss = self.heatmap_loss(predicted_flap_heatmaps, flap_heatmaps)
                loss += flap_loss

        # log corner keypoints to AP metric, frame by frame
        formatted_gt_corner_keypoints = [
            [Keypoint(int(k[0]), int(k[1])) for k in frame_corner_keypoints]
            for frame_corner_keypoints in corner_keypoints
        ]
        for i, predicted_corner_frame_heatmap in enumerate(torch.unbind(predicted_corner_heatmaps, 0)):
            detected_corner_keypoints = self.extract_detected_keypoints(predicted_corner_frame_heatmap)
            self.validation_metric.update(detected_corner_keypoints, formatted_gt_corner_keypoints[i])

        self.log("validation_loss", loss)

    def validation_epoch_end(self, outputs):
        ## called on the end of the validation epoch.
        ap = self.validation_metric.compute()
        print(f"{ap=}")
        self.log("validation_ap", ap)
        self.validation_metric.reset()

    ##################
    # util functions #
    ##################

    def create_heatmap_batch(self, shape: Tuple[int, int], keypoints: torch.Tensor) -> torch.Tensor:
        """[summary]

        Args:
            shape (Tuple): H,W
            keypoints (torch.Tensor): N x K x 3 Tensor with batch of keypoints.

        Returns:
            (torch.Tensor): N x H x W Tensor with N heatmaps
        """
        # TODO: profile to see if the conversion from and to GPU does not introduce a bottleneck
        # alternative is to create heatmaps on GPU by passing device to the generate_keypoints_heatmap function

        # convert keypoints to cpu to create the heatmaps
        batch_heatmaps = [
            generate_keypoints_heatmap(shape, keypoints[i].cpu(), self.heatmap_sigma) for i in range(len(keypoints))
        ]
        batch_heatmaps = np.stack(batch_heatmaps, axis=0)
        batch_heatmaps = torch.from_numpy(batch_heatmaps)
        return batch_heatmaps.to(self.device)

    def extract_detected_keypoints(self, heatmap: torch.Tensor) -> List[DetectedKeypoint]:
        """
        get keypoints of single channel from single frame.

        Args:
        heatmap (torch.Tensor) : B x H x W tensor that represents a heatmap.
        """

        detected_keypoints = get_keypoints_from_heatmap(heatmap, self.minimal_keypoint_pixel_distance)
        keypoint_probabilities = self.compute_keypoint_probability(heatmap, detected_keypoints)
        detected_keypoints = [
            DetectedKeypoint(detected_keypoints[i][0], detected_keypoints[i][1], keypoint_probabilities[i])
            for i in range(len(detected_keypoints))
        ]

        return detected_keypoints

    def compute_keypoint_probability(self, heatmap: torch.Tensor, detected_keypoints: List[List[int]]) -> List[float]:
        """Compute probability measure for each detected keypoint on the heatmap

        Args:
            heatmap (torch.Tensor): Heatmap
            detected_keypoints (List[List[int]]): List of extreacted keypoints

        Returns:
            List[float]: [description]
        """
        return [heatmap[k[0]][k[1]].item() for k in detected_keypoints]
