#!/bin/bash -e
D="fashion"
N="c2f1"
L=( pcontrast ptriplet pmtriplet pstriplet )
M=( E N C )

tmux new-session -s "$D" "gpustat -i 10" \; detach
for l in ${L[@]}; do
	for m in ${M[@]}; do
		tmux attach-session -t "$D" \; new-window -n "$D:$N:$l$m" \
			"CUDA_VISIBLE_DEVICES=0 python3 train.py -g1 -C $D:$N:$l$m; sh" \; detach
	done
done
