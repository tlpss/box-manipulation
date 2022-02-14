import argparse
import distutils.util

import timm
import torch.nn as nn

from keypoint_detection.models.backbones.backbone_factory import Backbone


class PretrainedBackbone(Backbone):
    def __init__(self, timm_model_name, freeze_weights):
        super().__init__()
        freeze_weights = distutils.util.strtobool(freeze_weights)  # convert argparse to bool

        assert timm_model_name in timm.list_models(pretrained=True)

        self.model: nn.Module = timm.create_model(timm_model_name, pretrained=True, pooling="", num_classes=0)

        if freeze_weights:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)

    def get_n_channels_out(self):
        return self.model.get_classifier().in_features

    @staticmethod
    def add_to_argparse(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("UnetBackbone")
        parser.add_argument("--timm_model_name", type=str, default="resnet-18")
        parser.add_argument("--freeze_pretrained_weights", default=False, type=str, required=False)
        return parent_parser


if __name__ == "__main__":
    model = PretrainedBackbone()
