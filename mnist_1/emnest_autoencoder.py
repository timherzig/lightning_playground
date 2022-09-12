import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import torch

class Emnest_AutoEncoder(pl.LightningModule):

    def __init__(self, batch_size, learning_rate):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28 * 28)
        )

        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        enc = self.encoder(x)
        dec = self.decoder(enc)
        loss = F.mse_loss(x, dec)
        self.log('trian_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        enc = self.encoder(x)
        dec = self.decoder(enc)
        loss = F.mse_loss(x, dec)
        self.log('valid_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        enc = self.encoder(x)
        dec = self.decoder(enc)
        loss = F.mse_loss(x, dec)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer