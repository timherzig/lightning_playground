#!/bin/bash

srun \
--container-image=/netscratch/enroot/dlcc_pytorch_20.10.sqsh \
--container-mounts=/ds/images/MNIST:/ds/MNIST \
--container-mounts=/netscratch/$USER/mnist_1:/cpt/mnist \
--container-mounts=/home/herzig/misc/lightning_playground/mnist_1:/src \
--container-workdir=/src \
python3 emnest_train.py -mnist /ds/MNIST -checkpoint /cpt/mnist
echo 'running'