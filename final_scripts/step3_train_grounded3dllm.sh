#!/bin/bash
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

NUM_GPUS=4
CURR_TOPK=750
CURR_QUERY=150

DATA=indoor_dialog
MODEL=mask3d_lang

# TRAIN
python main_run.py \
general.experiment_name="step3_${MODEL}_${NUM_GPUS}GPUS" \
general.project_name="scannet200" \
general.gpus=${NUM_GPUS} \
data=${DATA} \
model=${MODEL}  \
general.checkpoint=saved/step2_mask3d_lang_4GPUS/last-epoch.ckpt \
data.batch_size=5 \
data.num_workers=4 \
trainer=trainer50 \
optimizer.lr=0.0008 \
general.train_mode=true \
general.timestamp=$(date +"%m-%d-%H-%M-%S") \
general.filter_scene00=false \
general.topk_per_image=${CURR_TOPK} \
general.llm_config=conf/llm/tiny_vicuna_len512.json 

# TEST
python main_run.py \
general.experiment_name="step3_${MODEL}_${NUM_GPUS}GPUS" \
general.project_name="scannet200" \
general.gpus=${NUM_GPUS} \
data=${DATA} \
model=${MODEL}  \
data.batch_size=5 \
data.num_workers=4 \
trainer=trainer50 \
optimizer.lr=0.0008 \
general.train_mode=false \
general.timestamp=$(date +"%m-%d-%H-%M-%S") \
general.filter_scene00=false \
general.topk_per_image=${CURR_TOPK} \
general.llm_config=conf/llm/tiny_vicuna_len512.json 
