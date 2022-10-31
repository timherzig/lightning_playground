#!/bin/bash

srun \
--container-image=/netscratch/enroot/hufe_slt_dlcc_pytorch_20.10.sqsh.packages:pytorch-lightning==1.0.4 \
--container-mounts=/ds:/ds,/netscratch:/netscratch,"`pwd`":"`pwd`" \
--container-workdir="`pwd`" \
install.sh python3 emnest_train.py -mnist '/ds/images/MNIST' -checkpoint '/netscratch/herzig/mnist_1'
echo 'running'