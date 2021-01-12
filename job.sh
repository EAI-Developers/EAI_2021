#!/bin/sh
#$ -cwd
#$ -l long
# $ -l gpus=1
#$ -e ./logs/
#$ -o ./logs/
mkdir -p ./logs/
mkdir -p ./runs/
mkdir -p ./output/

CUDA_VISIBLE_DEVICES=2 \
python -m torchbeast.monobeast \
       # --env MsPacmanNoFrameskip-v4 \
       # --env KungFuNoFrameskip-v4 \
       # --env PongNoFrameskip-v4 \
       # --env BreakoutNoFrameskip-v4 \
       --env SeaquestNoFrameskip-v4 \
       # --num_actors 10 \
       # --num_threads 2 \
       --total_steps 2_000_000_000 \
       # --learning_rate 0.0002 \
       --grad_norm_clipping 1280 \
       --epsilon 0.01 \
       --entropy_cost 0.01 \
       --xpid "run_Seaquest" \
       --savedir "./output/output_Seaquest" \
       # --unroll_length 50 \
       # --batch_size 32 \
       --num_actors 60 \
       --unroll_length 160 \
       --num_buffers 80 \
       --num_threads 6 \
       --batch_size 4 \
       --learning_rate 0.001


