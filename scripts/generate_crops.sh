#!/usr/bin/env bash

#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_VISIBLE_DEVICES=3

PREPROCESS_NAME=$1
PROCS=$2
GEN_CROP=$3
NUM_TRAIN_SAMPLES=$4
CROP_SUB_PARAMS=$5

CROP_PARAMS="--in_path data/generated/graph_levels/${PREPROCESS_NAME}/train/graphs/ --out_path data/generated/cropped/${PREPROCESS_NAME}/train/graphs/ ${CROP_SUB_PARAMS}"

if [ "$GEN_CROP" = true ] ; then
  echo "Starting job crop with ${NUM_TRAIN_SAMPLES} samples over ${PROCS} processors"
  start=$SECONDS
  seq 0 $((NUM_TRAIN_SAMPLES-1)) | xargs -P $PROCS -I{} python preprocessing/crop_training_samples.py --number {} \
  $CROP_PARAMS
  duration=$(( SECONDS - start ))
  echo "Job duration: ${duration}"
fi
