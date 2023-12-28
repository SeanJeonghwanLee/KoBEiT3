#!/bin/bash

echo "### START DATE=$(date)"
echo "### HOSTNAME=$(hostname)"
echo "### CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
 
# conda 환경 활성화.
source  ~/.bashrc
conda   activate   svqa2
 
# cuda 11.0 환경 구성.
ml purge
ml load cuda/11.0
 
# 활성화된 환경에서 코드 실행.

python -m torch.distributed.launch --nproc_per_node=6 run_beit3_finetuning.py \
        --model beit3_base_patch16_384 \
        --input_size 384 \
        --task vqacustom \
        --batch_size 72 \
        --sentencepiece_model /home/seanlee/class/SpeechVQAPipeline/models/smp.model \
        --finetune /home/seanlee/class/SpeechVQAPipeline/beit3/finetune_checkpoint/checkpoint-best.pth \
        --data_path /home/seanlee/class/SpeechVQAPipeline/ \
        --output_dir /home/seanlee/class/SpeechVQAPipeline/beit3/results \
        --dist_eval

echo "###"
echo "### END DATE=$(date)"
