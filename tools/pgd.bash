##!/bin/bash
set -eux
PGD_PARAM_28='eps=0.3:pgditer=24'
if (echo $1 | grep -s -o _fashion-) \
        || (echo $1 | grep -s -o _mnist-); then
        python3 bin/advclass.py -v -A PGD:${PGD_PARAM_28} -C $@
else
        echo "??? Usage: $0 <model.ckpt> [args...]"
fi
