#!/bin/sh
SOURCE=$1
TARGET=$2
ARCH1=$3
ARCH2=$4
ARCH3=$5

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 main/target_train.py -dt ${TARGET} \
	--num-instances 4 --lr 0.00035 --iters 800 -b 64 --epochs 40 \
	--init-1 logs/${SOURCE}TO${TARGET}/${ARCH1}-pretrain/model_best.pth.tar \
	--init-2 logs/${SOURCE}TO${TARGET}/${ARCH2}-pretrain/model_best.pth.tar \
	--init-3 logs/${SOURCE}TO${TARGET}/${ARCH3}-pretrain/model_best.pth.tar \
	--logs-dir logs/${SOURCE}TO${TARGET}/${ARCH1}-${ARCH2}-${ARCH3}-MEB-Net
