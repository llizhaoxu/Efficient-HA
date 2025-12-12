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


python generate_response_chair_llava.py \
    --device "cuda:0" \
    --model_id "llava-hf/llava-1.5-7b-hf" \
    --output "/projects/_ssd/ZhaoxuCode/Efficient-HA/Result/output_beam.json" \
    --method "beam" \
    --max_tokens 512 \