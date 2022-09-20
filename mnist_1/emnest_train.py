import os
import torch

import pytorch_lightning as pl
import torchvision.transforms as transforms

from pl_bolts.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from emnest_autoencoder import Emnest_AutoEncoder

pl.seed_everything(1234)
batch_size = 4096

dataset = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())

mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())

mnist_train, mnist_val = random_split(dataset, [55000, 5000])

train_loader = DataLoader(mnist_train, batch_size=batch_size)
test_loader = DataLoader(mnist_test, batch_size=batch_size)
val_loader = DataLoader(mnist_val, batch_size=batch_size)


model = Emnest_AutoEncoder(batch_size=batch_size, learning_rate=1e-3)

trainer = pl.Trainer(accelerator='gpu', devices=[1], max_epochs=30)

print(f'Number of devices: {torch.cuda.device_count()}')
print(f'Device used: {torch.cuda.get_device_name(torch.cuda.current_device())}')

trainer.fit(model, train_loader, val_loader)