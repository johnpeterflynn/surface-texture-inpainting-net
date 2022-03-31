#!/usr/bin/env bash

#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_VISIBLE_DEVICES=3

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export PATH=$PATH:${SCRIPT_DIR}/../vcglib/apps/tridecimator
export PATH=$PATH:${SCRIPT_DIR}/../vcglib/apps/sample/trimesh_clustering

PREPROCESS_NAME=$1
PROCS=$2
GEN_TRAIN=$3
GEN_VAL=$4
NUM_TRAIN_SAMPLES=$5
NUM_VAL_SAMPLES=$6
GRAPH_SUB_PARAMS=$7

GRAPH_PARAMS="--in_path data/scannet/scans/ ${GRAPH_SUB_PARAMS} --dataset scannet" # --verbose_out_path output/
TRAIN_GRAPH_OUT_PATH="data/generated/graph_levels/${PREPROCESS_NAME}/train/graphs/"
VAL_GRAPH_OUT_PATH="data/generated/graph_levels/${PREPROCESS_NAME}/val/graphs/"

if [ "$GEN_TRAIN" = true ] ; then
  echo "Starting job gen train with ${NUM_TRAIN_SAMPLES} samples over ${PROCS} processors"
  start=$SECONDS
  seq 0 $((NUM_TRAIN_SAMPLES-1)) | xargs -P $PROCS -I{} python preprocessing/graph_level_generation.py --number {} \
  $GRAPH_PARAMS --out_path $TRAIN_GRAPH_OUT_PATH
  duration=$(( SECONDS - start ))
  echo "Job duration: ${duration}"
fi

if [ "$GEN_VAL" = true ] ; then
  echo "Starting job gen val with ${NUM_VAL_SAMPLES} samples over ${PROCS} processors"
  start=$SECONDS
  seq 0 $((NUM_VAL_SAMPLES-1)) | xargs -P $PROCS -I{} python preprocessing/graph_level_generation.py --number {} \
  $GRAPH_PARAMS --out_path  $VAL_GRAPH_OUT_PATH --val
  duration=$(( SECONDS - start ))
  echo "Job duration: ${duration}"
fi
