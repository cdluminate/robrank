#!/bin/bash
# Copyright (C) 2019-2022, Mo Zhou <cdluminate@gmail.com>
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -e
command -v tmux >/dev/null || ( echo "tmux not found"; exit 1 )

usage () {
	cat << EOF
swipe all checkpoints --
    specifically, given a set of usable cuda devices, we first glob
    all checkpoints from the given directory, and then spawn a
    swipe.py process for each discovered checkpoint using one of the
    usable cuda devices.
usage: $0 -p <profile> <directory>
note: 1. depends on tmux
      2. specify cuda visisble devices in advance
example:
      bash tools/swipeall.bash logs_cub-rres18p-ptripletN/ -p rob224 -a '-m 10'
sidenote:
  you may obtain multiple checkpoints with (--trail), e.g. :
    python3 bin/train.py -C cub:rres18p:ptripletN --trail
EOF
}
test -z "$1" && (usage; exit 0)

PROFILE=''
DIR=''
ARGS=''
while test -n "$1"; do
	case $1 in
		-p|--profile)
			PROFILE="$2"
			shift
			shift
			;;
		-h|--help)
			usage  
			exit 0
			;;
		-a|--arg)
			# additional argument for swipe.py
			ARGS="$2"
			shift
			shift
			;;
		*)
			if test -d "$1"; then
				DIR="${1}"
				shift
			else
				echo unrecognized argument: $1
				exit 3
			fi
			;;
	esac
done

test -z "${CUDA_VISIBLE_DEVICES}" && (echo "pleaseee specify cuda devices"; exit 2)
test -z "${PROFILE}" && (echo "please specify profile"; exit 4)
test -z "${DIR}" && (echo please specify directory; exit 5)

DEVMAP=( $(echo ${CUDA_VISIBLE_DEVICES} | sed -e 's/,/ /g') )
echo We can use ${DEVMAP[@]} devices.

CKPT=( $( find $DIR -path '*.ckpt' ) )
echo Found ${#CKPT[@]} checkpoints. Preparing to spawn tmux windows ...
#sleep 5

counter=1
for ckpt in ${CKPT[@]}; do
	idx=$(( (counter - 1) % ${#DEVMAP[@]} ))
	dev=${DEVMAP[$idx]}
#python3 bin/swipe.py -p rob224 -C logs_cub-rres18-ptripletN/.../xxx.ckpt
	echo [ID $counter DEV $dev: python3 bin/swipe.py -p ${PROFILE} -C ${ckpt} ${ARGS}
	tmux new-window -d -n "SwAll[$counter]" \
		"CUDA_VISIBLE_DEVICES=${dev} python3 bin/swipe.py -p ${PROFILE} -C ${ckpt} ${ARGS}; sh"
	true $((counter++))
done
