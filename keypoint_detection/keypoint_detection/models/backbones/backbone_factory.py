import argparse

from keypoint_detection.models.backbones.base_backbone import Backbone
from keypoint_detection.models.backbones.dilated_cnn import DilatedCnn
from keypoint_detection.models.backbones.s3k import S3K


class BackboneFactory:
    @staticmethod
    def create_backbone(backbone: str, **kwargs) -> Backbone:
        if backbone == "DilatedCnn":
            return DilatedCnn(**kwargs)
        elif backbone == "S3K":
            return S3K()
        else:
            raise Exception("Unknown backbone type")

    @staticmethod
    def add_to_argparse(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("BackboneFactory")
        parser.add_argument("--backbone", type=str, default="DilatedCnn")
        return parent_parser
