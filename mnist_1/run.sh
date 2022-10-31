#!/bin/bash

srun \
--container-image=/netscratch/enroot/dlcc_pytorch_20.10.sqsh \
--container-mounts=/ds/images/MNIST:/ds/MNIST,/netscratch/$USER/mnist_1:/cpt/mnist \
--container-workdir="`pwd`" \
install.sh python3 emnest_train.py -mnist '/ds/MNIST' -checkpoint '/cpt/mnist'
echo 'running'