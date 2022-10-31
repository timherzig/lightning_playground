#!/bin/bash

srun \
--container-image=/netscratch/enroot/dlcc_pytorch_20.10.sqsh \
--container-mounts=/ds:/ds,/netscratch:/netscratch,"`pwd`":"`pwd`" \
--container-workdir="`pwd`" \
install.sh python3 emnest_train.py -mnist '/ds/images/MNIST' -checkpoint '/netscratch/$USER/mnist_1'
echo 'running'