import torch

import pytorch_lightning as pl

from torchaudio.models import Emformer

class EmformerModel(pl.LightningModule):
    def __init__(self, in_d, n_heads, ffn_d, n_layers, seg_len, lr, bs):
        super().__init__()

        self.in_d = in_d
        self.n_heads = n_heads
        self.ffn_d = ffn_d
        self.n_layers = n_layers
        self.seg_len = seg_len

        self.bs = bs
        self.lr = lr

        self.emformer = Emformer(
            input_dim=self.in_d,
            num_heads=self.n_heads,
            ffn_dim=self.ffn_d,
            num_layers=self.n_layers,
            segment_length=self.seg_len,
            right_context_length=1
        )

    def forward(self, x):
        return self.emformer(x)

    def training_step(self, batch, batch_idx):
        x, x_lengths, y = batch
        o, o_lengths = self.emformer(x, x_lengths)
        ctc_loss = torch.nn.CTCLoss()
        loss = ctc_loss(o, y, o_lengths, len(y))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, x_lengths, y = batch
        o, o_lengths = self.emformer(x, x_lengths)
        ctc_loss = torch.nn.CTCLoss()
        loss = ctc_loss(o, y, o_lengths, len(y))
        self.log('valid_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, x_lengths, y = batch
        o, o_lengths = self.emformer(x, x_lengths)
        ctc_loss = torch.nn.CTCLoss()
        loss = ctc_loss(o, y, o_lengths, len(y))
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer