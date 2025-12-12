#!/bin/bash

#SBATCH --job-name=llava_greedy
#SBATCH --gres=gpu:a40:1     
#SBATCH --output=logs/log_%j.out         
#SBATCH --time=24:00:00
#SBATCH --qos rose



# NOTE:
# For a list of helpful flags to specify above, check out Slurm Overview in slurm.md.

module load Miniforge3  
source activate
conda activate EHA  


python generate_response_pope.py \
    --device "cuda:0" \
    --model_id "llava-hf/llava-1.5-7b-hf" \
    --output "/projects/_ssd/ZhaoxuCode/Efficient-HA/Result/POPE/llava-1.5-7b" \
    --pope_type "random"\
    --method "greedy" \
    --max_tokens 10 \

python generate_response_pope.py \
    --device "cuda:0" \
    --model_id "llava-hf/llava-1.5-7b-hf" \
    --output "/projects/_ssd/ZhaoxuCode/Efficient-HA/Result/POPE/llava-1.5-7b" \
    --pope_type "popular"\
    --method "greedy" \
    --max_tokens 10 \

python generate_response_pope.py \
    --device "cuda:0" \
    --model_id "llava-hf/llava-1.5-7b-hf" \
    --output "/projects/_ssd/ZhaoxuCode/Efficient-HA/Result/POPE/llava-1.5-7b" \
    --pope_type "adversarial"\
    --method "greedy" \
    --max_tokens 10 \

python generate_response_pope.py \
    --device "cuda:0" \
    --model_id "llava-hf/llava-1.5-7b-hf" \
    --output "/projects/_ssd/ZhaoxuCode/Efficient-HA/Result/POPE/llava-1.5-7b" \
    --pope_type "random"\
    --method "beam" \
    --max_tokens 10 \

python generate_response_pope.py \
    --device "cuda:0" \
    --model_id "llava-hf/llava-1.5-7b-hf" \
    --output "/projects/_ssd/ZhaoxuCode/Efficient-HA/Result/POPE/llava-1.5-7b" \
    --pope_type "popular"\
    --method "beam" \
    --max_tokens 10 \

python generate_response_pope.py \
    --device "cuda:0" \
    --model_id "llava-hf/llava-1.5-7b-hf" \
    --output "/projects/_ssd/ZhaoxuCode/Efficient-HA/Result/POPE/llava-1.5-7b" \
    --pope_type "adversarial"\
    --method "beam" \
    --max_tokens 10 \

python generate_response_pope.py \
    --device "cuda:0" \
    --model_id "llava-hf/llava-1.5-7b-hf" \
    --output "/projects/_ssd/ZhaoxuCode/Efficient-HA/Result/POPE/llava-1.5-7b" \
    --pope_type "random"\
    --method "dola" \
    --max_tokens 10 \

python generate_response_pope.py \
    --device "cuda:0" \
    --model_id "llava-hf/llava-1.5-7b-hf" \
    --output "/projects/_ssd/ZhaoxuCode/Efficient-HA/Result/POPE/llava-1.5-7b" \
    --pope_type "popular"\
    --method "dola" \
    --max_tokens 10 \

python generate_response_pope.py \
    --device "cuda:0" \
    --model_id "llava-hf/llava-1.5-7b-hf" \
    --output "/projects/_ssd/ZhaoxuCode/Efficient-HA/Result/POPE/llava-1.5-7b" \
    --pope_type "adversarial"\
    --method "dola" \
    --max_tokens 10 \


python generate_response_pope.py \
    --device "cuda:0" \
    --model_id "llava-hf/llava-1.5-7b-hf" \
    --output "/projects/_ssd/ZhaoxuCode/Efficient-HA/Result/POPE/llava-1.5-7b" \
    --pope_type "random"\
    --method "deco" \
    --max_tokens 10 \

python generate_response_pope.py \
    --device "cuda:0" \
    --model_id "llava-hf/llava-1.5-7b-hf" \
    --output "/projects/_ssd/ZhaoxuCode/Efficient-HA/Result/POPE/llava-1.5-7b" \
    --pope_type "popular"\
    --method "deco" \
    --max_tokens 10 \

python generate_response_pope.py \
    --device "cuda:0" \
    --model_id "llava-hf/llava-1.5-7b-hf" \
    --output "/projects/_ssd/ZhaoxuCode/Efficient-HA/Result/POPE/llava-1.5-7b" \
    --pope_type "adversarial"\
    --method "deco" \
    --max_tokens 10 \