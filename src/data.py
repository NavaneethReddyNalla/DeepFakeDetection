import os
import lightning as pl
import torch

from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from torch.utils.data import DataLoader


class DeepFakeDataModule(pl.LightningDataModule):
    def __init__(self, train_dir:str, valid_dir:str, test_dir:str, batch_size: int = 32):
        super(DeepFakeDataModule, self).__init__()
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.train_set = None
        self.valid_set = None
        self.test_set = None
        self.transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            self.train_set = ImageFolder(self.train_dir, self.transforms)
            self.valid_set = ImageFolder( self.valid_dir, self.transforms)

        if stage == "test" or stage is None:
            self.test_set = ImageFolder( self.test_dir, self.transforms)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.valid_set, batch_size=self.batch_size)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_set, batch_size=self.batch_size)
