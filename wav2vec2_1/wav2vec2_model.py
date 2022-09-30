import torch

import pytorch_lightning as pl

from jiwer import wer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Config


class W2V2Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.configuration = Wav2Vec2Config()

        self.wav2vec2 = Wav2Vec2ForCTC(config=self.configuration)
    
    def forward(self, **inputs):
        return self.wav2vec2(**inputs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        o = self.emformer(x)
        ctc_loss = torch.nn.CTCLoss()
        loss = ctc_loss(o, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        o = self.emformer(x)
        ctc_loss = torch.nn.CTCLoss()
        loss = ctc_loss(o, y)
        self.log('valid_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        o = self.wav2vec2(x)
        ctc_loss = torch.nn.CTCLoss()
        loss = ctc_loss(o, y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer