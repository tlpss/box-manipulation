import argparse
import distutils.util
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from keypoint_detection.models.backbones.backbone_factory import BackboneFactory
from keypoint_detection.models.metrics import DetectedKeypoint, Keypoint, KeypointAPMetrics
from keypoint_detection.models.utils.loss import LossFactory
from keypoint_detection.models.utils.visualization import visualize_predictions
from keypoint_detection.utils.heatmap import generate_keypoints_heatmap, get_keypoints_from_heatmap
from keypoint_detection.utils.tensor_padding import unpad_nans_from_tensor


class KeypointDetector(pl.LightningModule):
    """
    Box keypoint Detector using Gaussian Heatmaps
    There are 2 channels (groups) of keypoints, one that detects box corners and
    one that detects the center (or corners) of the outer edges of all flaps of the box.

    """

    @staticmethod
    def add_model_argparse_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        add named arguments from the init function to the parser
        The default values here are actually duplicates from the init function, but this was for readability (??)
        """
        parser = parent_parser.add_argument_group("KeypointDetector")

        # TODO: add these with inspection to avoid manual duplication!

        parser.add_argument("--heatmap_sigma", type=int, required=False)
        parser.add_argument("--detect_flap_keypoints", default=True, type=str, required=False)
        parser.add_argument("--minimal_keypoint_extract_pixel_distance", type=int, required=False)
        parser.add_argument("--maximal_gt_keypoint_pixel_distances", type=str, required=False)
        parser.add_argument("--learning_rate", type=float, required=False)

        BackboneFactory.add_to_argparse(parent_parser)
        LossFactory.add_to_argparse(parent_parser)

        return parent_parser

    def __init__(
        self,
        heatmap_sigma=10,
        detect_flap_keypoints: Union[bool, str] = True,
        maximal_gt_keypoint_pixel_distances: Union[str, List[float]] = None,
        minimal_keypoint_extraction_pixel_distance: int = None,
        learning_rate: float = 5e-4,
        backbone: str = "DilatedCnn",
        loss: str = "bce",
        **kwargs,
    ):
        """[summary]

        Args:
            heatmap_sigma (int, optional): Sigma of the gaussian heatmaps used to train the detector. Defaults to 10.
            n_channels (int, optional): Number of channels for the CNN layers. Defaults to 32.
            detect_flap_keypoints (bool, optional): Detect flap keypoints in a second channel or use a single channel Detector for box corners only.
            minimal_keypoint_extraction_pixel_distance (int, optional): the minimal distance (in pixels) between two detected keypoints,
                                                                        or the size of the local mask in which a keypoint needs to be the local maximum
            maximal_gt_keypoint_pixel_distance (int, optional): the maximal distance between a gt keypoint and detected keypoint, for the keypoint to be considered a TP
            kwargs: Pythonic catch for the other named arguments, used so that we can use a dict with ALL system hyperparameters to initialise the model from this
                    hyperparamater configuration dict. The alternative is to add a single 'hparams' argument to the init function, but this is imo less readable.
                    cf https://pytorch-lightning.readthedocs.io/en/stable/common/hyperparameters.html for an overview.
        """
        super().__init__()
        ## No need to manage devices ourselves, pytorch.lightning does all of that.
        ## device can be accessed through self.device if required.
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # to add new hyperparameters:
        # 1. define as named arg in the init (and use them)
        # 2. add to the argparse method of this module
        # 3. pass them along when calling the train.py file to override their default value
        self.learning_rate = learning_rate

        if isinstance(detect_flap_keypoints, str):
            detect_flap_keypoints = distutils.util.strtobool(detect_flap_keypoints)
        self.detect_flap_keypoints = detect_flap_keypoints

        self.heatmap_sigma = heatmap_sigma

        self.ap_epoch_start = 2
        self.ap_epoch_freq = 3

        if maximal_gt_keypoint_pixel_distances:

            # if str (from argparse, convert to list of ints)
            if isinstance(maximal_gt_keypoint_pixel_distances, str):
                maximal_gt_keypoint_pixel_distances = [
                    float(val) for val in maximal_gt_keypoint_pixel_distances.strip().split(" ")
                ]

            self.maximal_gt_keypoint_pixel_distances = maximal_gt_keypoint_pixel_distances
        else:
            self.maximal_gt_keypoint_pixel_distances = [heatmap_sigma]

        if minimal_keypoint_extraction_pixel_distance:
            self.minimal_keypoint_pixel_distance = int(minimal_keypoint_extraction_pixel_distance)
        else:
            self.minimal_keypoint_pixel_distance = int(min(self.maximal_gt_keypoint_pixel_distances))

        self.corner_validation_metric = KeypointAPMetrics(self.maximal_gt_keypoint_pixel_distances)

        if self.detect_flap_keypoints:
            self.flap_validation_metric = KeypointAPMetrics(self.maximal_gt_keypoint_pixel_distances)

        self.n_channels_out = (
            2 if self.detect_flap_keypoints else 1
        )  # number of keypoint classes = number of output channels of CNN

        # create model
        backbone = BackboneFactory.create_backbone(backbone, **kwargs)
        head = nn.Conv2d(
            in_channels=backbone.get_n_channels_out(),
            out_channels=self.n_channels_out,
            kernel_size=(3, 3),
            padding="same",
        )
        self.model = nn.Sequential(
            backbone,
            head,
            nn.Sigmoid(),  # create probabilities
        )

        self.heatmap_loss = LossFactory.create_loss(loss, **kwargs)
        # save hyperparameters to logger, to make sure the model hparams are saved even if
        # they are not included in the config (i.e. if they are kept at the defaults).
        # this is for later reference and consistency.
        self.save_hyperparameters(ignore="**kwargs")

    def forward(self, x: torch.Tensor):
        """
        x shape must be (N,C_in,H,W) with N batch size, and C_in number of incoming channels (3)
        return shape = (N, 1, H,W)
        """
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.learning_rate)
        return optimizer

    def shared_step(self, batch, batch_idx, validate=False) -> Dict[str, Any]:
        """
        shared step for training, validation (and testing)
        computes heatmaps and loss for corners and flaps (if self.detect_flap_keypoints is True)

        returns:

        shared_dict (Dict): a dict with the heatmaps, gt_keypoints and losses
        """
        imgs, corner_keypoints, flap_keypoints = batch
        corner_keypoints = [unpad_nans_from_tensor(corner_keypoints[i, :]) for i in range(corner_keypoints.shape[0])]
        flap_keypoints = [unpad_nans_from_tensor(flap_keypoints[i, :]) for i in range(flap_keypoints.shape[0])]
        # load here to device to keep mem consumption low, if possible one could also load entire dataset on GPU to speed up training..
        imgs = imgs.to(self.device)

        ## predict and compute losses
        corner_heatmaps = self.create_heatmap_batch(imgs[0].shape[1:], corner_keypoints)
        predicted_heatmaps = self.forward(imgs)  # create heatmaps JIT, is this desirable?
        predicted_corner_heatmaps = predicted_heatmaps[:, 0, :, :]
        corner_loss = self.heatmap_loss(predicted_corner_heatmaps, corner_heatmaps)
        loss = corner_loss
        gt_corner_loss = self.heatmap_loss(corner_heatmaps, corner_heatmaps)
        gt_loss = gt_corner_loss
        result_dict = {
            "corner_loss": corner_loss.detach(),
            "corner_keypoints": corner_keypoints,
        }

        # only pass predictions in validate step to avoid overhead in train step.
        if validate:
            result_dict.update({"predicted_heatmaps": predicted_heatmaps.detach()})

        if self.detect_flap_keypoints:
            flap_heatmaps = self.create_heatmap_batch(imgs[0].shape[1:], flap_keypoints)
            predicted_flap_heatmaps = predicted_heatmaps[:, 1, :, :]
            flap_loss = self.heatmap_loss(predicted_flap_heatmaps, flap_heatmaps)
            loss = loss + flap_loss  # cannot do inline on tensor with grad!
            gt_flap_loss = self.heatmap_loss(flap_heatmaps, flap_heatmaps)
            gt_loss = gt_loss + gt_flap_loss
            result_dict.update({"flap_loss": flap_loss.detach(), "flap_keypoints": flap_keypoints})

        result_dict.update({"loss": loss, "gt_loss": gt_loss})

        # visualization
        if batch_idx == 0 and self.current_epoch > 0:
            visualize_predictions(
                imgs,
                predicted_corner_heatmaps.detach(),
                corner_heatmaps,
                self.logger,
                self.minimal_keypoint_pixel_distance,
                validate=validate,
            )
            if self.detect_flap_keypoints:
                visualize_predictions(
                    imgs,
                    predicted_flap_heatmaps.detach(),
                    flap_heatmaps,
                    self.logger,
                    self.minimal_keypoint_pixel_distance,
                    keypoint_class="flap",
                    validate=validate,
                )

        return result_dict

    def training_step(self, train_batch, batch_idx):

        result_dict = self.shared_step(train_batch, batch_idx)

        # logging
        self.log("train/corner_loss", result_dict["corner_loss"])
        self.log("train/loss", result_dict["loss"])
        self.log("train/gt_loss", result_dict["gt_loss"])
        if self.detect_flap_keypoints:
            self.log("train/flap_loss", result_dict["flap_loss"])
        return result_dict

    def validation_step(self, val_batch, batch_idx):

        result_dict = self.shared_step(val_batch, batch_idx, validate=True)

        if self.is_ap_epoch():
            # update corner AP metric
            predicted_corner_heatmaps = result_dict["predicted_heatmaps"][:, 0, :, :]
            gt_corner_keypoints = result_dict["corner_keypoints"]
            self.update_ap_metrics(predicted_corner_heatmaps, gt_corner_keypoints, self.corner_validation_metric)

            if self.detect_flap_keypoints:
                predicted_flap_heatmaps = result_dict["predicted_heatmaps"][:, 1, :, :]
                gt_flap_keypoints = result_dict["flap_keypoints"]
                self.update_ap_metrics(predicted_flap_heatmaps, gt_flap_keypoints, self.flap_validation_metric)

        ## log (defaults to on_epoch, which aggregates the logged values over entire validation set)
        self.log("validation/epoch_loss", result_dict["loss"])
        self.log("validation/gt_loss", result_dict["gt_loss"])

    def validation_epoch_end(self, outputs):
        """
        Called on the end of the validation epoch.
        Used to compute and log the AP metrics.
        """

        if self.is_ap_epoch():
            mean_ap = self.compute_and_log_metrics(self.corner_validation_metric)

            if self.detect_flap_keypoints:
                flap_ap = self.compute_and_log_metrics(self.flap_validation_metric, "flap")
                mean_ap = (mean_ap + flap_ap) / 2

            self.log("meanAP", mean_ap)

    ##################
    # util functions #
    ##################
    @classmethod
    def get_artifact_dir_path(cls) -> Path:
        return Path(__file__).resolve().parents[2] / "artifacts"

    @classmethod
    def get_wand_log_dir_path(cls) -> Path:
        return Path(__file__).resolve().parents[2] / "wandb"

    def update_ap_metrics(
        self, predicted_heatmaps: torch.Tensor, gt_keypoints: torch.Tensor, validation_metric: KeypointAPMetrics
    ):
        """
        Update provided AP metric by extracting the detected keypoints for each heatmap
        and combining them with the gt keypoints for the same frame
        """
        # log corner keypoints to AP metrics, frame by frame
        formatted_gt_keypoints = [
            [Keypoint(int(k[0]), int(k[1])) for k in frame_gt_keypoints] for frame_gt_keypoints in gt_keypoints
        ]
        for i, predicted_frame_heatmap in enumerate(torch.unbind(predicted_heatmaps, 0)):
            detected_corner_keypoints = self.extract_detected_keypoints(predicted_frame_heatmap)
            validation_metric.update(detected_corner_keypoints, formatted_gt_keypoints[i])

    def compute_and_log_metrics(self, validation_metric: KeypointAPMetrics, keypoint_class: str = "corner") -> float:
        """
        logs ap for each max_distance, resets metric and returns meanAP
        """
        # compute ap's
        ap_metrics = validation_metric.compute()
        print(f"{ap_metrics=}")
        for maximal_distance, ap in ap_metrics.items():
            self.log(f"validation/{keypoint_class}_ap/d={maximal_distance}", ap)

        mean_ap = sum(ap_metrics.values()) / len(ap_metrics.values())

        self.log(f"validation/{keypoint_class}_meanAP", mean_ap)  # log top level for wandb hyperparam chart.
        validation_metric.reset()
        return mean_ap

    def is_ap_epoch(self):
        return self.ap_epoch_start <= self.current_epoch and self.current_epoch % self.ap_epoch_freq == 0

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

    def compute_keypoint_probability(
        self, heatmap: torch.Tensor, detected_keypoints: List[Tuple[int, int]]
    ) -> List[float]:
        """Compute probability measure for each detected keypoint on the heatmap

        Args:
            heatmap (torch.Tensor): Heatmap
            detected_keypoints (List[List[int]]): List of extreacted keypoints

        Returns:
            List[float]: [description]
        """
        return [heatmap[k[0]][k[1]].item() for k in detected_keypoints]
