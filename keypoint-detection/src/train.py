import os
import sys

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from src.datamodule import BoxKeypointsDataModule, BoxKeypointsDataset
from src.models import KeypointDetector

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

print(sys.path)

IMAGE_DATASET_PATH = "/workspaces/box-manipulation/datasets/box_dataset2"
JSON_PATH = "/workspaces/box-manipulation/datasets/box_dataset2/dataset.json"

pl.seed_everything(2021, workers=True)  # deterministic run
model = KeypointDetector()
module = BoxKeypointsDataModule(BoxKeypointsDataset(JSON_PATH, IMAGE_DATASET_PATH), 2)


wandb_logger = WandbLogger(project="test-project", entity="airo-box-manipulation")
trainer = pl.Trainer(max_epochs=2, logger=wandb_logger, gpus=0)
trainer.fit(model, module)
