#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`

EXP_DIR=exps/multibaseline
OUTPUT_DIR=output/tzb_multi
mkdir -p ${EXP_DIR}
PY_ARGS=${@:1}
python -u main.py \
    --test \
    --backbone swin_b_p4w7 \
    --epochs 6 \
    --num_feature_levels 1 \
    --num_queries 100 \
    --hidden_dim 256 \
    --dilation \
    --batch_size 1 \
    --num_ref_frames 14 \
    --resume ${EXP_DIR}/checkpoint0005.pth \
    --lr_drop_epochs 4 5 \
    --num_workers 0 \
    --with_box_refine \
    --dataset_file 'tzb_multi' \
    --output_dir ${OUTPUT_DIR} \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.test_e6.$T