export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 run_beit3_finetuning.py \
        --model beit3_large_indomain_patch16_224_vqacustom \
        --input_size 224 \
        --task vqacustom \
        --batch_size 72 \
        --sentencepiece_model /home/seanlee/class/SpeechVQAPipeline/models/smp.model \
        --finetune /home/seanlee/class/SpeechVQAPipeline/beit3/finetune_checkpoint/checkpoint-best.pth \
        --data_path /scratch/seanlee/ \
        --output_dir /home/seanlee/class/SpeechVQAPipeline/beit3/results \
        --dist_eval