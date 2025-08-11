#!/bin/bash

MODEL="HDiT-XL/2"
PER_PROC_BATCH_SIZE=256
NUM_FID_SAMPLES=50000
PATH_TYPE="linear"
MODE="sde"
NUM_STEPS=250
# cfg_scale=1.0 (without CFG),by default we use cfg_scale=1.8 with guidance interval.
CFG_SCALE=1.0
GUIDANCE_HIGH=1.0
RESOLUTION=256
VAE="ema"
GLOBAL_SEED=0
SAMPLE_DIR="exps/latent-XL2/fid-samples"
CKPT="exps/latent-XL2/checkpoints/step-350000.pt"

# Multi-GPU distributed launch (original). Uncomment to use DDP with 8 GPUs.
# python -m torch.distributed.launch \
#     --nproc_per_node=8 \
#     generate.py \
#     --num-fid-samples $NUM_FID_SAMPLES \
#     --path-type $PATH_TYPE \
#     --per-proc-batch-size $PER_PROC_BATCH_SIZE \
#     --mode $MODE \
#     --num-steps $NUM_STEPS \
#     --cfg-scale $CFG_SCALE \
#     --guidance-high $GUIDANCE_HIGH \
#     --sample-dir $SAMPLE_DIR \
#     --model $MODEL \
#     --ckpt $CKPT \
#     --vae $VAE \
#     --resolution $RESOLUTION \
#     --global-seed $GLOBAL_SEED \

# Single GPU sampling (no DDP)
python generate.py \
    --no-ddp \
    --num-fid-samples $NUM_FID_SAMPLES \
    --path-type $PATH_TYPE \
    --per-proc-batch-size $PER_PROC_BATCH_SIZE \
    --mode $MODE \
    --num-steps $NUM_STEPS \
    --cfg-scale $CFG_SCALE \
    --guidance-high $GUIDANCE_HIGH \
    --sample-dir $SAMPLE_DIR \
    --model $MODEL \
    --ckpt $CKPT \
    --vae $VAE \
    --resolution $RESOLUTION \
    --global-seed $GLOBAL_SEED \


python npz_convert.py \
     --model $MODEL \
    --ckpt $CKPT \
    --sample-dir $SAMPLE_DIR \
    --num-fid-samples $NUM_FID_SAMPLES \
    --resolution $RESOLUTION \
    --vae $VAE \
    --cfg-scale $CFG_SCALE \
    --global-seed $GLOBAL_SEED \
    --mode $MODE \




