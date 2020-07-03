#!/bin/sh
TARGET=$1
ARCH=$2
MODEL=$3

CUDA_VISIBLE_DEVICES=15 \
python main/model_test.py -b 256 -j 8 \
	--dataset-target ${TARGET} -a ${ARCH} --resume ${MODEL}
