#!/usr/bin/env bash

rm -rf temp_vr1
torchrun --standalone \
    --nnodes=1 \
    --nproc-per-node=4 \
    src/marvin_recipes/scripts/verify_data_resumption.py \
    --tmp_dir temp_vr1 \
    --seed=31 \
    --epochs=7 \
    --checkpoint_steps=800 \
    --checkpoint_dir="train_checkpoints" \
    --recover_from="train_checkpoints/step_3200" \
    --model_chk="meta-llama/Llama-2-7b-hf" \
    --data_blend_path="data_blends/marvin_blend_01.json"