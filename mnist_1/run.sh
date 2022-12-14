#!/bin/bash

srun \
--gpus=1 \
--container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_22.10-py3.sqsh \
--container-mounts=/ds:/ds,/netscratch:/netscratch,"`pwd`":"`pwd`" \
--container-workdir="`pwd`" \
install.sh python3 emnest_train.py -mnist '/ds/images/MNIST' -checkpoint '/netscratch/herzig/mnist_1'
echo 'running'