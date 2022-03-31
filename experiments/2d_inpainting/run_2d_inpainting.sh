#!/usr/bin/env bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

python train.py -c experiments/2d_inpainting/config/config_stinet_imageinpainting.json  -n "image_inpainting_final" -g "dummy_hash" --ld "saved/final" -m "Final results for inpainting using STINet or Resnet2D equivalent"
