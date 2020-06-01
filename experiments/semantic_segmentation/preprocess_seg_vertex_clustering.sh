#!/usr/bin/env bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3

GEN_TRAIN=true
GEN_VAL=true
GEN_CROP=true

PROCS=12
NUM_TRAIN_SAMPLES=1201
NUM_VAL_SAMPLES=312

#NOTE: From DualConvMeshNet paper, vertex clustering works better than qem for segmentation
#PREPROCESS_NAME="vc_only_indices_undialated"
#GRAPH_SUB_PARAMS="--vertex_clustering --level_params 0.04 0.08 0.16 0.32 --dilated_levels 0 0 0 0"
PREPROCESS_NAME="qem_indices_undialated"
GRAPH_SUB_PARAMS="--qem --level_params 100 30 30 30 --dilated_levels 0 0 0 0"
CROP_SUB_PARAMS="--block_size 3.0 --stride 1.5"

./scripts/generate_graph_levels.sh "$PREPROCESS_NAME" "$PROCS" "$GEN_TRAIN" "$GEN_VAL" "$NUM_TRAIN_SAMPLES" "$NUM_VAL_SAMPLES" "$GRAPH_SUB_PARAMS"
./scripts/generate_crops.sh "$PREPROCESS_NAME" "$PROCS" "$GEN_CROP" "$NUM_TRAIN_SAMPLES" "$CROP_SUB_PARAMS"
