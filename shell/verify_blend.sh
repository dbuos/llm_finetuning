#!/usr/bin/env bash

rm -rf temp_vr1
torchrun --standalone \
    --nnodes=1 \
    --nproc-per-node=4 \
    src/marvin_recipes/scripts/verify_dist_blend.py \
    --temp_base temp_vr1 \
    --workers=8 \
    --data_blend_path="data_blends/marvin_blend_01_choose.json"
#    --data_blend_path="data_blends/marvin_blend_01.json"