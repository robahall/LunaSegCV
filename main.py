from argparse import ArgumentParser
import pytorch_lightning as pl
from src.data.dataset import LitLuna
from src.models.luna_cnn import LunaClassCNN

def main():
    luna_dataset = LitLuna(batch_size=16)
    model = LunaClassCNN()
    trainer = pl.Trainer()
    trainer.fit(model, datamodule=luna_dataset)

if __name__ == "__main__":
    main()


