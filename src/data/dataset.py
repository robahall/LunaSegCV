import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from src.data.utils import LunaDataset, getCandidateInfoList
from torchvision import transforms

class LitLuna(pl.LightningDataModule):

    def __init__(self, batch_size):
        super().__init__()
        self.transform = transforms.Compose([
        ])
        self.batch_size = batch_size

    def prepare_data(self):
        pass

    def setup(self):
        self.train= LunaDataset(candidateInfo_list=getCandidateInfoList(), valid=False)
        self.val = LunaDataset(candidateInfo_list=getCandidateInfoList(), valid=True)


    def train_dataloader(self) -> DataLoader:
        transforms = self.transform
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        transforms = self.transform
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        transforms = self.transform
        return DataLoader(self.test, batch_size=self.batch_size)




