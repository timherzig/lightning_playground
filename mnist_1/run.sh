#!/bin/bash

srun \
--container-image=/netscratch/enroot/dlcc_pytorch_20.10.sqsh.packages:pytorch-lightning==1.0.4 \
--container-workdir=/home/$USER/misc/lightning_playground/mnist_1:/src \
--container-mounts=/ds/images/MNIST:/ds/MNIST \
--container-mounts=/netscratch/$USER/mnist_1:/cpt/mnist \
python3 /src/emnest_train.py -mnist /ds/MNIST -checkpoint /cpt/mnist
echo 'running'