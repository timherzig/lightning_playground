import torch
import pytorch_lightning as pl

from lib.parser import parse_arguments
from data.commonvoice_dataset import CommonVoiceDataset
from data.batch_transform import pad_batch

from torch.utils.data import DataLoader
from wav2vec2_model import W2V2Model


def main():
    args = parse_arguments()

    wav2vec2model = W2V2Model()

    train_dataset = CommonVoiceDataset('train', args.cv_dir, n_rows=100)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=pad_batch)

    val_dataset = CommonVoiceDataset('dev', args.cv_dir, n_rows=10)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=pad_batch)

    trainer = pl.Trainer(accelerator='gpu', devices=[1], max_epochs=1, default_root_dir=args.log_dir)

    trainer.fit(wav2vec2model, train_loader, val_loader)

    return


if __name__ == '__main__':
    main()