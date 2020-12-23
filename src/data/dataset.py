import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from src.data.utils import LunaDataset, getCandidateInfoList
from torchvision import transforms

class LitLuna(pl.LightningDataModule):

    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose([
        ])
        self.train_dims = None

    def prepare_data(self):
        pass

    def setup(self):
        self.train= LunaDataset(candidateInfo_list=getCandidateInfoList(), valid=False)
        self.val = LunaDataset(candidateInfo_list=getCandidateInfoList(), valid=True)
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




