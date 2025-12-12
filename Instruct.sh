#!/bin/bash

#SBATCH --job-name=llava_greedy
#SBATCH --gres=gpu:a40:1     
#SBATCH --output=logs/log_%j.out         
#SBATCH --time=12:00:00
#SBATCH --qos rose



# NOTE:
# For a list of helpful flags to specify above, check out Slurm Overview in slurm.md.

module load Miniforge3  
source activate
conda activate EHA  
nvidia-smi

# CUDA_VISIBLE_DEVICES=0 python generate_response_chair_llava.py \
#     --model_id "Salesforce/instructblip-vicuna-7b" \
#     --output "/projects/_ssd/ZhaoxuCode/Efficient-HA/Result/Instructblip/output_greedy.json" \
#     --method "greedy" \
#     --max_tokens 512 

CUDA_VISIBLE_DEVICES=0 python generate_response_chair_llava.py \
    --model_id "Salesforce/instructblip-vicuna-7b" \
    --output "/projects/_ssd/ZhaoxuCode/Efficient-HA/Result/Instructblip/output_beam.json" \
    --method "beam" \
    --max_tokens 512  

