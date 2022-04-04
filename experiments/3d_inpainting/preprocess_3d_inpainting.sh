#!/usr/bin/env bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

GEN_TRAIN=true
GEN_VAL=true
GEN_CROP=true

PROCS=12
NUM_TRAIN_SAMPLES=1201
NUM_VAL_SAMPLES=312

PREPROCESS_NAME="qem_indices_dilated_2_4_6_8_16_circlemask"
GRAPH_SUB_PARAMS="--qem --level_params 100 30 30 30 --dilated_levels 0 0 1 0 --dilation_dists 2 4 6 8 16"
CROP_SUB_PARAMS="--block_size 3.0 --stride 1.5"

NUM_SAMPLES=$((NUM_TRAIN_SAMPLES+NUM_VAL_SAMPLES))
MASK_NAME="rad_16"
MASK_SUB_PARAMS="circles --radius 16 --frac_masked_vertices 0.25 --masks_per_scene 16"

./scripts/generate_graph_levels.sh "$PREPROCESS_NAME" "$PROCS" "$GEN_TRAIN" "$GEN_VAL" "$NUM_TRAIN_SAMPLES" "$NUM_VAL_SAMPLES" "$GRAPH_SUB_PARAMS"
./scripts/generate_crops.sh "$PREPROCESS_NAME" "$PROCS" "$GEN_CROP" "$NUM_TRAIN_SAMPLES" "$CROP_SUB_PARAMS"
./scripts/generate_masks.sh "$PREPROCESS_NAME" "$MASK_NAME" "$PROCS" "$NUM_SAMPLES" "$MASK_SUB_PARAMS"