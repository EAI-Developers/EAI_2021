#!/bin/sh
#$ -cwd
#$ -l long
#$ -l gpus=1
#$ -e ./logs/
#$ -o ./logs/
mkdir -p ./logs/
mkdir -p ./runs/
mkdir -p ./output/

# MsPacmanNoFrameskip
# BreakoutNoFrameskip
# KungFuMasterNoFrameskip
# PongNoFrameskip
# SeaquestNoFrameskip

CUDA_VISIBLE_DEVICES=2 python -m torchbeast.monobeast \
       --env SeaquestNoFrameskip-v4 \
       --xpid cpu \
       --num_actors 60\
       --num_buffers 80\
       --num_threads 6 \
       --total_steps 2_000_000_000 \
       --learning_rate 0.001 \
       --grad_norm_clipping 1280 \
       --epsilon 0.01 \
       --entropy_cost 0.01 \
       --xpid "run_Seaquest" \
       --savedir "./output/Seaquest" \
       --unroll_length 60 --batch_size 4

