#!/bin/bash

EXP_DIR=outputs1/exp_antiuav_transvod_lite_swinb
mkdir -p ${EXP_DIR}

python -u main.py \
  --dataset_file vid_multi \
  --vid_path a_Ncode/output \
  --output_dir ${EXP_DIR} \
  --with_box_refine \
  --two_stage \
  --num_queries 300 \
  --epochs 50 \
  --lr 2e-4 \
  --lr_backbone 1e-5 \
  --batch_size 1 \
  --backbone swin_b \
  --window_size 8 \
  --num_feature_levels 4 \
  --enc_layers 6 \
  --dec_layers 6 \
  --dim_feedforward 2048 \
  --hidden_dim 256 \
  --dropout 0.1 \
  --nheads 8 \
  --position_embedding sine \
  --pretrained_weights exps/our_models/COCO_pretrained_model/swinb.pth \
  --seed 42 \
  --eval
