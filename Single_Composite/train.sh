#!/usr/bin/env bash
CONFIG=$1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node=4 \
  --master_port=4321 \
  -m basicsr.train \
  -opt $CONFIG \
  --launcher pytorch
