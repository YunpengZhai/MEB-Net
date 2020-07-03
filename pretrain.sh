#!/bin/sh
SOURCE=$1
TARGET=$2
ARCH=$3

CUDA_VISIBLE_DEVICES=0,1 \
python3 main/source_pretrain.py -ds ${SOURCE} -dt ${TARGET} -a ${ARCH} --margin 0.0 \
	--num-instances 4 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 200 --epochs 80 --eval-step 5 \
	--logs-dir logs/${SOURCE}TO${TARGET}/${ARCH}-pretrain