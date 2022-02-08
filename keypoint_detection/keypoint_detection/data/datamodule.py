import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class BoxKeypointsDataModule(pl.LightningDataModule):
    def __init__(self, dataset: Dataset, batch_size: int = 4, validation_split_ratio=0.1):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

        validation_size = int(validation_split_ratio * len(self.dataset))
        train_size = len(self.dataset) - validation_size
        self.train_dataset, self.validation_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, validation_size]
        )

    def train_dataloader(self):
        dataloader = DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=4)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(self.validation_dataset, self.batch_size, shuffle=False, num_workers=4)
        return dataloader

    def test_dataloader(self):
        pass
