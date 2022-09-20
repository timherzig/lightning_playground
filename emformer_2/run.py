import yaml

import torch
import pytorch_lightning as pl

from lib.parser import parse_arguments
from torch.utils.data import DataLoader
from emformer_model import Emformer_Model
from data.batch_transform import pad_batch
from data.commonvoice_dataset import CommonVoiceDataset


def main():
    args = parse_arguments()

    with open(args.model_config, 'r') as stream:
        try:
            model_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    emformer_model = Emformer_Model(model_config)

    if args.mode == 'train':
        train_dataset = CommonVoiceDataset('train', '/home/tim/Documents/dfki/datasets/cv-corpus-10.0-2022-07-04-de/cv-corpus-10.0-2022-07-04/de/', n_rows=100)
        train_loader = DataLoader(train_dataset, batch_size=model_config['bs'], shuffle=True, num_workers=model_config['nw'], collate_fn=pad_batch)

        val_dataset = CommonVoiceDataset('dev', '/home/tim/Documents/dfki/datasets/cv-corpus-10.0-2022-07-04-de/cv-corpus-10.0-2022-07-04/de/', n_rows=10)
        val_loader = DataLoader(val_dataset, batch_size=model_config['bs'], shuffle=True, num_workers=model_config['nw'], collate_fn=pad_batch)

        trainer = pl.Trainer(accelerator='gpu', devices=[1], max_epochs=1)

        trainer.fit(emformer_model, train_loader, val_loader)

        print(f'Number of devices: {torch.cuda.device_count()}')
        print(f'Device used: {torch.cuda.get_device_name(torch.cuda.current_device())}')

if __name__ == "__main__":
    main()