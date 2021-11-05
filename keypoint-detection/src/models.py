import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from src.keypoint_utils import generate_keypoints_heatmap


class KeypointDetector(pl.LightningModule):
    # TODO: REFACTOR & CLEAN
    """
    Box keypoint Detector using Gaussian Heatmaps
    There are 2 channels (groups) of keypoints, one that detect box corners and once that detects the center of the outer edge of all flaps of the box.

    """

    def __init__(self, heatmap_sigma=10):
        super().__init__()
        ## No need to manage devices ourselves, pytorch.lightning does all of that.
        ## device can be accessed through self.device if required.
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.heatmap_sigma = heatmap_sigma

        n_channels = 32
        n_channels_in = 3
        n_channels_out = 1  # number of keypoint classes
        kernel_size = (3, 3)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=n_channels_in, out_channels=n_channels, kernel_size=kernel_size, padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, dilation=2, padding="same"
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, dilation=4, padding="same"
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, dilation=8, padding="same"
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, dilation=16, padding="same"
            ),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, dilation=2, padding="same"
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, dilation=4, padding="same"
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, dilation=8, padding="same"
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, dilation=16, padding="same"
            ),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels_out, kernel_size=kernel_size, padding="same"),
            nn.Sigmoid(),
        ).to(self.device)

    def forward(self, x):
        """
        x shape must be (N,C_in,H,W) with N batch size, and C_in number of incoming channels (3)
        return shape = (N, 1, H,W)
        """
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def loss(self, predicted_heatmaps, heatmaps):
        # No focal loss as in CenterNet paper bc @Peter said it does not improve performance too much, so KISS and maybe add it later
        return torch.nn.functional.binary_cross_entropy(predicted_heatmaps, heatmaps, reduction="mean")

    def create_heatmap_batch(self, shape, keypoints):
        # TODO: profile to see if the conversion from and to GPU does not introduce a bottleneck
        # alternative is to create heatmaps on GPU by passing device to the generate_keypoints_heatmap function

        # convert keypoints to cpu to create the heatmaps
        batch_heatmaps = [
            generate_keypoints_heatmap(shape, keypoints[i].cpu(), self.heatmap_sigma) for i in range(len(keypoints))
        ]
        batch_heatmaps = np.stack(batch_heatmaps, axis=0)
        batch_heatmaps = torch.from_numpy(batch_heatmaps)
        return batch_heatmaps.to(self.device)

    def training_step(self, train_batch, batch_idx):
        imgs, corner_keypoints, flap_keypoints = train_batch
        # load here to device to keep mem consumption low, if possible one could also load entire dataset on GPU to speed up training..
        imgs = imgs.to(self.device)
        # create heatmaps JIT, is this desirable?
        corner_heatmaps = self.create_heatmap_batch(imgs[0].shape[1:], corner_keypoints)
        # flap_heatmaps = self.create_heatmap_batch(img[0].shape[1:], flap_keypoints)

        predicted_heatmaps = self.forward(imgs)
        predicted_corner_heatmaps = predicted_heatmaps[:, 0, :, :]

        corner_loss = self.loss(predicted_corner_heatmaps, corner_heatmaps)

        loss = corner_loss

        ## logging
        self.log("train_corner_loss", corner_loss)
        self.log("train_loss", loss)
        return loss
