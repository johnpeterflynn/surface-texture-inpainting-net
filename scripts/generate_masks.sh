#!/usr/bin/env bash

#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_VISIBLE_DEVICES=0

PREPROCESS_NAME=$1
MASK_NAME=$2
PROCS=$3
NUM_SAMPLES=$4
SUB_PARAMS=$5

PARAMS="--in_path ./data/scannet/scans --mask_name ${MASK_NAME} --preprocess_name ${PREPROCESS_NAME}"
#SUB_PARAMS="observers --in_sens_path ./data/scannet_sens_data"
#SUB_PARAMS="circles --radius 16 --frac_masked_vertices 0.25 --masks_per_scene 16"
OUT_PATH="output/circles"

echo "Starting job gen train with ${NUM_SAMPLES} samples over ${PROCS} processors"
start=$SECONDS
seq 0 $((NUM_SAMPLES-1)) | xargs -P $PROCS -I{} python preprocessing/observed_texture_map_generation.py --number {} \
$PARAMS --out_path $OUT_PATH $SUB_PARAMS
duration=$(( SECONDS - start ))
echo "Job duration: ${duration}"