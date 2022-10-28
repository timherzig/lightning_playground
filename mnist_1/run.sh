#!/bin/bash

srun \
--container-image=/netscratch/enroot/dlcc_pytorch_20.10.sqsh.packages:pytorch-lightning==1.0.4 \
--container-mounts=/home/$USER/misc/lightning_playground/mnist_1:/src/mnist \
--container-mounts=/ds/images/MNIST:/ds/MNIST \
--container-workdir=/netscratch/$USER/mnist_1:/cpt/mnist \
python3 /src/mnist/emnest_train.py -mnist /ds/MNIST -checkpoint /cpt/mnist