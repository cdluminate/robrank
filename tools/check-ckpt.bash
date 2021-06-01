#!/bin/bash
set -eux
PGD_PARAM_28='eps=0.30196:alpha=0.011764:pgditer=32'
PGD_PARAM_224='eps=0.03137:alpha=0.011764:pgditer=32'

function ckpt_sanity_check() {
if (echo $1 | grep -s -o _cub-) \
	|| (echo $1 | grep -s -o _cars-); then
	python3 advrank.py -A ES:${PGD_PARAM_224} -C $1 -m1
elif (echo $1 | grep -s -o _sop-); then
	python3 advrank.py -A ES:${PGD_PARAM_224} -C $1 -m1
elif (echo $1 | grep -s -o _mnist-) \
	|| (echo $1 | grep -s -o _fashion-); then
	python3 advrank.py -A ES:${PGD_PARAM_28} -C $1 -m1
else
	echo "??? Usage: $0 <model.ckpt>"
fi
}

CKPTS=( $(find . -path './logs_*/*.ckpt' -print) )
for ckpt in ${CKPTS[@]}; do
	echo ${ckpt}
	ckpt_sanity_check ${ckpt}
done
