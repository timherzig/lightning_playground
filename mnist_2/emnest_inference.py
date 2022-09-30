import os
import pytorch_lightning as pl
import torchvision.transforms as transforms

from pl_bolts.datasets import MNIST
from torch import optim, nn, utils, Tensor
from torch.utils.data import DataLoader, random_split
from emnest_autoencoder import Emnest_AutoEncoder

batch_size = 32

mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(mnist_test, batch_size=batch_size)

checkpoint = "lightning_logs/version_2/checkpoints/epoch=29-step=51570.ckpt"
model = Emnest_AutoEncoder.load_from_checkpoint(checkpoint, batch_size=32, learning_rate=1e-3)

encoder = model.encoder
encoder.eval()

fake_image_batch = Tensor(4, 28 * 28)
embeddings = encoder(fake_image_batch)
print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)
