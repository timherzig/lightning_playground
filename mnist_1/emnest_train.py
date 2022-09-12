import os
import torch

import pytorch_lightning as pl
import torchvision.transforms as transforms

from pl_bolts.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from emnest_autoencoder import Emnest_AutoEncoder

pl.seed_everything(1234)
batch_size = 32

dataset = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())

mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())

mnist_train, mnist_val = random_split(dataset, [55000, 5000])

train_loader = DataLoader(mnist_train, batch_size=batch_size)
test_loader = DataLoader(mnist_test, batch_size=batch_size)
val_loader = DataLoader(mnist_val, batch_size=batch_size)


model = Emnest_AutoEncoder(batch_size=32, learning_rate=1e-3)

trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=30)

trainer.fit(model, train_loader, val_loader)