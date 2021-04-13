#!/bin/sh -e

# === Fashion + C2F1 ==========================================================

# Defensive

tmux new-window "CUDA_VISIBLE_DEVICES=0 python3 train.py -C fashion:c2f1d:ptripletN ; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=2 python3 train.py -C fashion:c2f1d:pmtripletN; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=4 python3 train.py -C fashion:c2f1d:pstripletN; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=6 python3 train.py -C fashion:c2f1d:pdtripletN; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=0 python3 train.py -C fashion:c2f1d:pcontrastN; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=2 python3 train.py -C fashion:c2f1d:pgliftE   ; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=4 python3 train.py -C fashion:c2f1d:pnpairE   ; bash";

tmux new-window "CUDA_VISIBLE_DEVICES=0 python3 swipe.py -v -ppami_mnist -m8 -c logs_fashion-c2f1d-ptripletN ; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=2 python3 swipe.py -v -ppami_mnist -m8 -c logs_fashion-c2f1d-pmtripletN; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=4 python3 swipe.py -v -ppami_mnist -m8 -c logs_fashion-c2f1d-pstripletN; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=6 python3 swipe.py -v -ppami_mnist -m8 -c logs_fashion-c2f1d-pdtripletN; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=0 python3 swipe.py -v -ppami_mnist -m8 -c logs_fashion-c2f1d-pcontrastN; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=2 python3 swipe.py -v -ppami_mnist -m8 -c logs_fashion-c2f1d-pgliftE   ; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=4 python3 swipe.py -v -ppami_mnist -m8 -c logs_fashion-c2f1d-pnpairE   ; bash";

# EmbShift Suppression

tmux new-window "CUDA_VISIBLE_DEVICES=0 python3 train.py -C fashion:c2f1e:ptripletN ; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=2 python3 train.py -C fashion:c2f1e:pmtripletN; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=4 python3 train.py -C fashion:c2f1e:pstripletN; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=6 python3 train.py -C fashion:c2f1e:pdtripletN; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=0 python3 train.py -C fashion:c2f1e:pcontrastN; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=2 python3 train.py -C fashion:c2f1e:pgliftE   ; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=4 python3 train.py -C fashion:c2f1e:pnpairE   ; bash";

tmux new-window "CUDA_VISIBLE_DEVICES=0 python3 swipe.py -v -ppami_mnist -m8 -c logs_fashion-c2f1e-ptripletN ; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=2 python3 swipe.py -v -ppami_mnist -m8 -c logs_fashion-c2f1e-pmtripletN; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=4 python3 swipe.py -v -ppami_mnist -m8 -c logs_fashion-c2f1e-pstripletN; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=6 python3 swipe.py -v -ppami_mnist -m8 -c logs_fashion-c2f1e-pdtripletN; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=0 python3 swipe.py -v -ppami_mnist -m8 -c logs_fashion-c2f1e-pcontrastN; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=2 python3 swipe.py -v -ppami_mnist -m8 -c logs_fashion-c2f1e-pgliftE   ; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=4 python3 swipe.py -v -ppami_mnist -m8 -c logs_fashion-c2f1e-pnpairE   ; bash";

# Vanilla

tmux new-window "CUDA_VISIBLE_DEVICES=0 python3 train.py -C fashion:c2f1:ptripletN ; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=2 python3 train.py -C fashion:c2f1:pmtripletN; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=4 python3 train.py -C fashion:c2f1:pstripletN; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=6 python3 train.py -C fashion:c2f1:pdtripletN; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=0 python3 train.py -C fashion:c2f1:pcontrastN; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=2 python3 train.py -C fashion:c2f1:pgliftE   ; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=4 python3 train.py -C fashion:c2f1:pnpairE   ; bash";

tmux new-window "CUDA_VISIBLE_DEVICES=0 python3 swipe.py -v -ppami_mnist -m8 -c logs_fashion-c2f1-ptripletN ; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=2 python3 swipe.py -v -ppami_mnist -m8 -c logs_fashion-c2f1-pmtripletN; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=4 python3 swipe.py -v -ppami_mnist -m8 -c logs_fashion-c2f1-pstripletN; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=6 python3 swipe.py -v -ppami_mnist -m8 -c logs_fashion-c2f1-pdtripletN; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=0 python3 swipe.py -v -ppami_mnist -m8 -c logs_fashion-c2f1-pcontrastN; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=2 python3 swipe.py -v -ppami_mnist -m8 -c logs_fashion-c2f1-pgliftE   ; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=4 python3 swipe.py -v -ppami_mnist -m8 -c logs_fashion-c2f1-pnpairE   ; bash";

# === CUB + ResNet18 ==========================================================

# Defensive

# EmbShift Suppression
tmux new-window "CUDA_VISIBLE_DEVICES=0 python3 train.py -C cub:res18e:ptripletC ; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=2 python3 train.py -C cub:res18e:ptripletN ; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=4 python3 train.py -C cub:res18e:ptripletE ; bash";

# Vanilla

tmux new-window "CUDA_VISIBLE_DEVICES=0 python3 train.py -C cub:res18:ptripletN ; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=2 python3 train.py -C cub:res18:pmtripletN; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=4 python3 train.py -C cub:res18:pstripletN; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=6 python3 train.py -C cub:res18:pdtripletN; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=0 python3 train.py -C cub:res18:pcontrastN; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=2 python3 train.py -C cub:res18:pgliftE   ; bash";
tmux new-window "CUDA_VISIBLE_DEVICES=4 python3 train.py -C cub:res18:pnpairE   ; bash";

#

tmux new-window -n 'cub:res18:pmtripletE' "CUDA_VISIBLE_DEVICES=7 python3 train.py -C cub:res18:pmtripletE; bash";
tmux new-window -n 'cub:res18:pstripletE' "CUDA_VISIBLE_DEVICES=6 python3 train.py -C cub:res18:pstripletE; bash";
tmux new-window -n 'cub:res18:pmtripletN' "CUDA_VISIBLE_DEVICES=5 python3 train.py -C cub:res18:pmtripletN; bash";
tmux new-window -n 'cub:res18:pdtripletN' "CUDA_VISIBLE_DEVICES=4 python3 train.py -C cub:res18:pdtripletN; bash";

tmux new-window -n 'cub:enb0:pmtripletE' "CUDA_VISIBLE_DEVICES=7 python3 train.py -C cub:enb0:pmtripletE; bash";
tmux new-window -n 'cub:enb0:pstripletE' "CUDA_VISIBLE_DEVICES=6 python3 train.py -C cub:enb0:pstripletE; bash";
tmux new-window -n 'cub:enb0:pmtripletN' "CUDA_VISIBLE_DEVICES=5 python3 train.py -C cub:enb0:pmtripletN; bash";
tmux new-window -n 'cub:enb0:pstripletN' "CUDA_VISIBLE_DEVICES=4 python3 train.py -C cub:enb0:pstripletN; bash";

tmux new-window -n 'cub:res18d:ptripletC ' "CUDA_VISIBLE_DEVICES=0 python3 train.py -C cub:res18d:ptripletC ; bash";
tmux new-window -n 'cub:res18d:ptripletN ' "CUDA_VISIBLE_DEVICES=1 python3 train.py -C cub:res18d:ptripletN ; bash";
tmux new-window -n 'cub:res18d:ptripletE ' "CUDA_VISIBLE_DEVICES=2 python3 train.py -C cub:res18d:ptripletE ; bash";
tmux new-window -n 'cub:res18d:pdtripletN' "CUDA_VISIBLE_DEVICES=3 python3 train.py -C cub:res18d:pdtripletN; bash";

tmux new-window -n 'sop:res18d:ptripletC ' "CUDA_VISIBLE_DEVICES=4 python3 train.py -C sop:res18d:ptripletC ; bash";
tmux new-window -n 'sop:res18d:ptripletN ' "CUDA_VISIBLE_DEVICES=5 python3 train.py -C sop:res18d:ptripletN ; bash";
tmux new-window -n 'sop:res18d:ptripletE ' "CUDA_VISIBLE_DEVICES=6 python3 train.py -C sop:res18d:ptripletE ; bash";
tmux new-window -n 'sop:res18d:pdtripletN' "CUDA_VISIBLE_DEVICES=7 python3 train.py -C sop:res18d:pdtripletN; bash";

