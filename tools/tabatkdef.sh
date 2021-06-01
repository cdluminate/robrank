tmux new-window "CUDA_VISIBLE_DEVICES=0 python3 swipe.py -p pami28  -c logs_mnist-rc2f2p-ptripletN; sh"
tmux new-window "CUDA_VISIBLE_DEVICES=0 python3 swipe.py -p pami28  -c logs_mnist-rc2f2d-ptripletN; sh"
tmux new-window "CUDA_VISIBLE_DEVICES=0 python3 swipe.py -p pami28  -c logs_mnist-rc2f2-ptripletN; sh"

tmux new-window "CUDA_VISIBLE_DEVICES=1 python3 swipe.py -p pami28  -c logs_fashion-rc2f2p-ptripletN; sh"
tmux new-window "CUDA_VISIBLE_DEVICES=1 python3 swipe.py -p pami28  -c logs_fashion-rc2f2d-ptripletN; sh"
tmux new-window "CUDA_VISIBLE_DEVICES=1 python3 swipe.py -p pami28  -c logs_fashion-rc2f2-ptripletN; sh"

tmux new-window "CUDA_VISIBLE_DEVICES=2 python3 swipe.py -p pami224 -c logs_sop-rres18-ptripletN; sh"

tmux new-window "CUDA_VISIBLE_DEVICES=3 python3 swipe.py -p pami224 -c logs_sop-rres18d-ptripletN; sh"

tmux new-window "CUDA_VISIBLE_DEVICES=4 python3 swipe.py -p pami224 -c logs_sop-rres18p-ptripletN; sh"

tmux new-window "CUDA_VISIBLE_DEVICES=6 python3 swipe.py -p pami224 -c logs_cars-rres18p-ptripletN; sh"
tmux new-window "CUDA_VISIBLE_DEVICES=6 python3 swipe.py -p pami224 -c logs_cars-rres18d-ptripletN; sh"
tmux new-window "CUDA_VISIBLE_DEVICES=6 python3 swipe.py -p pami224 -c logs_cars-rres18-ptripletN; sh"

tmux new-window "CUDA_VISIBLE_DEVICES=7 python3 swipe.py -p pami224 -c logs_cub-rres18p-ptripletN; sh"
tmux new-window "CUDA_VISIBLE_DEVICES=7 python3 swipe.py -p pami224 -c logs_cub-rres18d-ptripletN; sh"
tmux new-window "CUDA_VISIBLE_DEVICES=7 python3 swipe.py -p pami224 -c logs_cub-rres18-ptripletN; sh"
