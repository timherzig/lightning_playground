import torch
import pytorch_lightning as pl

from torchaudio.models import Emformer

class Emformer_Model(pl.LightningModule):

    def __init__(self, model_args) -> None:
        super().__init__()

        self.in_d = model_args['emformer']['input_dim']
        self.n_heads = model_args['emformer']['num_heads']
        self.ffn_d = model_args['emformer']['ffn_dim']
        self.n_layers = model_args['emformer']['num_layers']
        self.seg_len = model_args['emformer']['segment_length']

        self.bs = model_args['bs']
        self.lr = model_args['lr']

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