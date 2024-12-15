import yaml
import os
import pytorch_lightning as pl
import torch

from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from torch.utils.data import DataLoader


class DeepFakeDataModule(pl.LightningDataModule):
    def __init__(self, data_dir:str = "", batch_size: int = 32, valid_set_eq_test_set: bool = False):
        super(DeepFakeDataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.valid_eq_test_set = valid_set_eq_test_set
        self.train_set = None
        self.valid_set = None
        self.test_set = None
        self.transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def setup(self, stage: str):
        test_path = "test" if not self.valid_eq_test_set else "valid"

        if stage == "fit" or stage is None:
            self.train_set = ImageFolder(os.path.join(self.data_dir, "train"), self.transforms)
            self.valid_set = ImageFolder(os.path.join(self.data_dir, "valid"), self.transforms)

        if stage == "test" or stage is None:
            self.test_set = ImageFolder(os.path.join(self.data_dir, test_path), self.transforms)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.valid_set, batch_size=self.batch_size)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_set, batch_size=self.batch_size)
