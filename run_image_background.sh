#!/bin/bash
dir_to_mount=$(pwd)/finetuning_repo
wandb_key=$(cat wandb_key.txt)
hf_token=$(cat hf_token.txt)
docker run -it --ipc=host -d --gpus all -v $HOME/.cache:/root/.cache -v $dir_to_mount:/workspace -e WANDB_API_KEY=$wandb_key -e HF_TOKEN=$hf_token gpt python /workspace/trl_finetune.py --block_size 1024 --eval_steps 100 --save_steps 100 --log_steps 100 -tf train.csv -vf validation.csv -m meta-llama/Llama-2-7b-hf  -b 1 -lr 2e-4 -e 1.0 --gradient_accumulation_steps 16 --pad_token_id=18610 --split_model -s 1
