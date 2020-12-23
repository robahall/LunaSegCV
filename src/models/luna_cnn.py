from torch import nn
from torch.nn import functional as F
from torch.optim import SGD
import pytorch_lightning as pl

class LunaClassCNN(pl.LightningModule):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()
        self.tail_batchnorm = nn.BatchNorm3d(1)
        self.block1 = LunaCNNBlock(in_channels, conv_channels)
        self.block2 = LunaCNNBlock(conv_channels, conv_channels*2)
        self.block3 = LunaCNNBlock(conv_channels*2, conv_channels*4)
        self.block4 = LunaCNNBlock(conv_channels*4, conv_channels*8)
        self.head_linear = nn.Linear(1152, 2)
        self.head_softmax = nn.Softmax(dim=1)

    def forward(self, X):
        bn_output = self.tail_batchnorm(X)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)

        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.head_linear(conv_flat)
        return self.head_softmax(linear_output)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def configure_optimizers(self):
        return SGD(self.model.parameters(), lr=0.001, momentum=0.99)


class LunaCNNBlock(pl.LightningModule):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, conv_channels, kernel_size=3, padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(conv_channels, conv_channels, kernel_size=3, padding=1, bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool =nn.MaxPool3d(2,2)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)
        return self.maxpool(block_out)
