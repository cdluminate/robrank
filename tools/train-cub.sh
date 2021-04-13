#!/bin/bash -e
D="cub"
N="res18"
L=( pcontrast ptriplet pmtriplet pstriplet )
M=( E N C )

echo tmux new-session -s "$D" "gpustat -i 10" \; detach
for l in ${L[@]}; do
	for m in ${M[@]}; do
		echo tmux attach-session -t "$D" \; new-window -n "$D:$N:$l$m" \
			"CUDA_VISIBLE_DEVICES=0 python3 train.py -g1 -C $D:$N:$l$m; sh" \; detach
	done
done

tmux new-session -s cub gpustat -i 10 \; detach
tmux attach-session -t cub \; new-window -n "cub:res18:pcontrastE" "CUDA_VISIBLE_DEVICES=0 python3 train.py -g1 -C cub:res18:pcontrastE; sh" \; detach
tmux attach-session -t cub \; new-window -n "cub:res18:pcontrastN" "CUDA_VISIBLE_DEVICES=0 python3 train.py -g1 -C cub:res18:pcontrastN; sh" \; detach
tmux attach-session -t cub \; new-window -n "cub:res18:pcontrastC" "CUDA_VISIBLE_DEVICES=2 python3 train.py -g1 -C cub:res18:pcontrastC; sh" \; detach
tmux attach-session -t cub \; new-window -n "cub:res18:ptripletE " "CUDA_VISIBLE_DEVICES=2 python3 train.py -g1 -C cub:res18:ptripletE;  sh" \; detach
tmux attach-session -t cub \; new-window -n "cub:res18:ptripletN " "CUDA_VISIBLE_DEVICES=3 python3 train.py -g1 -C cub:res18:ptripletN;  sh" \; detach
tmux attach-session -t cub \; new-window -n "cub:res18:ptripletC " "CUDA_VISIBLE_DEVICES=3 python3 train.py -g1 -C cub:res18:ptripletC;  sh" \; detach
tmux attach-session -t cub \; new-window -n "cub:res18:pmtripletE" "CUDA_VISIBLE_DEVICES=5 python3 train.py -g1 -C cub:res18:pmtripletE; sh" \; detach
tmux attach-session -t cub \; new-window -n "cub:res18:pmtripletN" "CUDA_VISIBLE_DEVICES=5 python3 train.py -g1 -C cub:res18:pmtripletN; sh" \; detach
tmux attach-session -t cub \; new-window -n "cub:res18:pmtripletC" "CUDA_VISIBLE_DEVICES=6 python3 train.py -g1 -C cub:res18:pmtripletC; sh" \; detach
tmux attach-session -t cub \; new-window -n "cub:res18:pstripletE" "CUDA_VISIBLE_DEVICES=6 python3 train.py -g1 -C cub:res18:pstripletE; sh" \; detach
tmux attach-session -t cub \; new-window -n "cub:res18:pstripletN" "CUDA_VISIBLE_DEVICES=7 python3 train.py -g1 -C cub:res18:pstripletN; sh" \; detach
tmux attach-session -t cub \; new-window -n "cub:res18:pstripletC" "CUDA_VISIBLE_DEVICES=7 python3 train.py -g1 -C cub:res18:pstripletC; sh" \; detach

