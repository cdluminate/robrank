#!/usr/bin/bash -e
dataset=fashion
model=c2f1

# LIST OF LOSS FUNCTIONS
L=(
#pcontrastE
pcontrastN
#pcontrastC
#pdcontrastN
#ptripletE
ptripletN
#ptripletC
#pmtripletE
pmtripletN
#pmtripletC
#pstripletE
pstripletN
#pstripletC
pdtripletN
pgliftE
pnpairE
#pmarginE
pmarginN
#pmarginC
#pdmarginN
#pquadE
pquadN
#pquadC
#pdquadN
pmsC
#pmsN
#prhomE
prhomN
#prhomC
#pdrhomN
)

# LIST OF JOBS
J=()
for l in ${L[@]}; do
	J+=( ${dataset}:${model}:${l} )
done
LOG=()
for i in ${L[@]}; do
	LOG+=( logs_${dataset}-${model}-${i} )
done


# LIST OF CARD IDS
C=(
0
1
2
3
4
5
6
7
)

# GENERATE SHELL CODE (TMUX)
c=0
for ((i=0; i<${#J[@]}; i++)) do
	cuda=${C[$(($i % ${#C[@]}))]}
	echo -n tmux new-window -n "$(printf "%27s" "\"${J[$i]}")\"" ' ';
	echo "\"CUDA_VISIBLE_DEVICES=${cuda} python3 train.py -C ${J[$i]}; bash\"";
done

# GENERATE SHELL CODE (TQ)
for ((i=0; i<${#J[@]}; i++)) do
	echo "tq r5 -- python3 train.py -C ${J[$i]}"
done
for i in ${LOG[@]}; do
	echo "tq r5 -- python3 swipe.py -v -ppami_mnist -m8 -c ${i}"
done
