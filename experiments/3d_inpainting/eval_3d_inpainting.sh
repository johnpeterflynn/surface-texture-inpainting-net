#!/usr/bin/env bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

python train.py -c experiments/3d_inpainting/config/config_stinet_surfacetextureinpainting.json  -n "surface_texture_inpainting_final" -g "dummy_hash" --ld "saved/final" -m "Final results for STINet on 3D ScanNet scenes" --eval "valid" --vis --resume ""
