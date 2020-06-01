#!/usr/bin/env bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# Segmentaton net
python train.py -c experiments/semantic_segmentation/config/config_scmnet_segmentation.json  -n "segmentation_singleconvmesnnet" -g "dummy_hash" --ld "saved/segmentation" -m "Training semantic segmentation network (SingleConvMeshNet) on ScanNet 3D scenes"
