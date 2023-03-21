#!/bin/sh
set -e
config="sop:rres18ghmetsm:ptripletN"
ckpt="logs_sop-rres18ghmetsm-ptripletN/lightning_logs/version_0/checkpoints/epoch=74-step=39900.ckpt"

# sometimes the communication through NCCL will make training hang.
export NCCL_P2P_DISABLE=1

# training in distributed data parallel
export CUDA_VISIBLE_DEVICES=0,1
python3 bin/train.py -C ${config}

# evaluation (only support single GPU mode)
export CUDA_VISIBLE_DEVICES=0
python3 bin/swipe.py -p rob224 -C ${ckpt}

# TODO
