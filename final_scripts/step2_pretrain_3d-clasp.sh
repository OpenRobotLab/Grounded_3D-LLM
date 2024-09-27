#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine
export TOKENIZERS_PARALLELISM=false

NUM_GPUS=4
CURR_TOPK=750
CURR_QUERY=150

DATA=indoor_grounding
MODEL=mask3d_lang

PRETRAINED=./saved/step1_mask3d_clip_4GPUS/last-epoch.ckpt

EXPNAME="step2_${MODEL}_${NUM_GPUS}GPUS"
echo "Running experiment name: ${EXPNAME}"

# TRAIN
HYDRA_FULL_ERROR=1 python main_run.py \
general.experiment_name="$EXPNAME" \
general.project_name="scannet200" \
general.gpus=${NUM_GPUS} \
data=$DATA \
model=$MODEL \
general.checkpoint=${PRETRAINED} \
data.batch_size=5 \
optimizer.lr=0.0001 \
trainer=trainerx2

# TEST
HYDRA_FULL_ERROR=1 python main_run.py \
general.gpus=${NUM_GPUS} \
general.experiment_name="$EXPNAME" \
general.project_name="scannet200" \
data=$DATA \
model=$MODEL \
model.num_queries=${CURR_QUERY} \
general.topk_per_image=${CURR_TOPK} \
general.use_dbscan=true  \
general.train_mode=false \
general.save_visualizations=false

