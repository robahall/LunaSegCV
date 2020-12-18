import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

class LitLuna(pl.LightningDataModule):

    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([
        ])
        self.train_dims = None
        self.vocab_size = 0

    def prepare_data(self):
        pass

    def setup(self):
        self.train, self.val, self.test = load_datasets()
        self.train_dims = self.train.next_batch.size()

    def train_dataloader(self) -> DataLoader:
        transforms = None
        return DataLoader(self.train, batch_size=64)

    def val_dataloader(self):
        transforms = None
        return DataLoader(self.val, batch_size=64)

    def test_dataloader(self):
        transforms = None
        return DataLoader(self.test, batch_size=64)




