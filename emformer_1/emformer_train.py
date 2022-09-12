from ast import main
import torch

import pytorch_lightning as pl

from torch.utils.data import DataLoader
from emformer_model import EmformerModel
from data.batch_transform import pad_batch
from data.commonvoice_dataset import CommonVoiceDataset

BATCH_SIZE = 1
NUM_WORKERS = 1

IN_D = 32
N_HEADS = 1
FFN_D = 64
N_LAYERS = 1
SEQ_LEN = 32
LR = 1e-3

if __name__ == "__main__":
    emformer = EmformerModel(IN_D, N_HEADS, FFN_D, N_LAYERS, SEQ_LEN, LR, BATCH_SIZE)

    train_dataset = CommonVoiceDataset('train', '/home/tim/Documents/datasets/cv-corpus-10.0-2022-07-04-de/cv-corpus-10.0-2022-07-04/de/', n_rows=100)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=pad_batch)

    val_dataset = CommonVoiceDataset('dev', '/home/tim/Documents/datasets/cv-corpus-10.0-2022-07-04-de/cv-corpus-10.0-2022-07-04/de/', n_rows=10)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=pad_batch)

    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=1)

    trainer.fit(emformer, train_loader, val_loader)

    # for i, data in enumerate(train_loader):
    #     speech_array, audio_len, transcription = data

    #     outputs = emformer

    #     if i == 0: break