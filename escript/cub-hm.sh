#!/bin/sh
set -e
config="cub:rres18ghmetsm:ptripletN"
ckpt="logs_cub-rres18ghmetsm-ptripletN/lightning_logs/version_0/checkpoints/epoch=74-step=3975.ckpt"

# -- sometimes the communication through NCCL will make training hang.
export NCCL_P2P_DISABLE=1

# -- training in distributed data parallel
export CUDA_VISIBLE_DEVICES=0,1
python3 bin/train.py -C ${config}

# -- evaluation (only support single GPU mode)
export CUDA_VISIBLE_DEVICES=0
python3 bin/swipe.py -p rob224 -C ${ckpt}
# a json file will be left besides the model checkpoint

# -- print the evaluation results
export ITH=0  # the n-to-the-last subdirectories.
# for instance, if there is only
# logs_sop-rres18ghmetsm-ptripletN/lightning_logs/version_0/
# then ITH=0 specifies the version_0 subdirectory.
# If there are both version_0 and version_1 subdirectories,
# ITH=0 specifies version_1, while ITH=1 specifies version_0.
# The pjswipe.py script will select version based on ITH.
# If you do not export ITH, the latest (one with the largest
# version number will be used)
python3 tools/pjswipe.py logs_cub-rres18ghmetsm-ptripletN
