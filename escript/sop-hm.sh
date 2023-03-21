export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
python3 bin/train.py -C sop:rres18ghmetsm:ptripletN
python3 bin/swipe.py -p rob224 -C logs_sop-rres18ghmetsm-ptripletN/lightning_logs/version_0/checkpoints/epoch=74-step=39900.ckpt
