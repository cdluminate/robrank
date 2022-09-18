#!/bin/bash
set -eux
PGD_PARAM_28='eps=0.30196:alpha=0.011764:pgditer=32'
PGD_PARAM_224='eps=0.03137:alpha=0.011764:pgditer=32'
pm=$1
shift
if (echo $1 | grep -s -o _cub-) \
	|| (echo $1 | grep -s -o _cars-); then
	python3 bin/advrank.py -v -A SPQA:pm=${pm}:M=1:${PGD_PARAM_224} -C $@
elif (echo $1 | grep -s -o _sop-); then
	python3 bin/advrank.py -v -A SPQA:pm=${pm}:M=1:${PGD_PARAM_224} -C $@
elif (echo $1 | grep -s -o _mnist-) \
	|| (echo $1 | grep -s -o _fashion-); then
	python3 bin/advrank.py -v -A SPQA:pm=${pm}:M=1:${PGD_PARAM_28} -C $@
else
	echo "??? Usage: $0 <+/-> <model.ckpt> [args...]"
fi
