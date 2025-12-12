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


# python generate_response_chair_llava.py \
#     --device "cuda:0" \
#     --model_id "llava-hf/llava-1.5-7b-hf" \
#     --output "/projects/_ssd/ZhaoxuCode/Efficient-HA/Result/output_dola.json" \
#     --method "dola" \
#     --max_tokens 512 \

# python generate_response_chair_llava.py \
#     --device "cuda:0" \
#     --model_id "llava-hf/llava-1.5-7b-hf" \
#     --output "/projects/_ssd/ZhaoxuCode/Efficient-HA/Result/output_deco.json" \
#     --method "deco" \
#     --max_tokens 512 \

# python generate_response_chair_llava.py \
#     --device "cuda:0" \
#     --model_id "Salesforce/instructblip-vicuna-7b" \
#     --output "/projects/_ssd/ZhaoxuCode/Efficient-HA/Result/InstructBlip/output_dola.json" \
#     --method "dola" \
#     --max_tokens 512 \

# python generate_response_chair_llava.py \
#     --device "cuda:0" \
#     --model_id "Salesforce/instructblip-vicuna-7b" \
#     --output "/projects/_ssd/ZhaoxuCode/Efficient-HA/Result/InstructBlip/output_deco.json" \
#     --method "deco" \
#     --max_tokens 512 \

# python generate_response_chair_llava.py \
#     --device "cuda:0" \
#     --model_id "Qwen/Qwen2.5-VL-7B-Instruct" \
#     --output "/projects/_ssd/ZhaoxuCode/Efficient-HA/Result/Qwen2.5-VL-7B/output_greedy.json" \
#     --method "greedy" \
#     --max_tokens 512 \

# python generate_response_chair_llava.py \
#     --device "cuda:0" \
#     --model_id "Qwen/Qwen2.5-VL-7B-Instruct" \
#     --output "/projects/_ssd/ZhaoxuCode/Efficient-HA/Result/Qwen2.5-VL-7B/output_beam.json" \
#     --method "beam" \
#     --max_tokens 512 \

# python generate_response_chair_llava.py \
#     --device "cuda:0" \
#     --model_id "Qwen/Qwen2.5-VL-7B-Instruct" \
#     --output "/projects/_ssd/ZhaoxuCode/Efficient-HA/Result/Qwen2.5-VL-7B/output_dola.json" \
#     --method "dola" \
#     --max_tokens 512 \

# python generate_response_chair_llava.py \
#     --device "cuda:0" \
#     --model_id "Qwen/Qwen2.5-VL-7B-Instruct" \
#     --output "/projects/_ssd/ZhaoxuCode/Efficient-HA/Result/Qwen2.5-VL-7B/output_deco.json" \
#     --method "deco" \
#     --max_tokens 512 \


# python generate_response_chair_llava.py \
#     --device "cuda:0" \
#     --model_id "Qwen/Qwen2.5-VL-7B-Instruct" \
#     --output "/projects/_ssd/ZhaoxuCode/Efficient-HA/Result/Qwen2.5-VL-7B/output_ours.json" \
#     --method "ours" \
#     --max_tokens 512 \


# python generate_response_chair_llava.py \
#     --device "cuda:0" \
#     --model_id "OpenGVLab/InternVL3-8B" \
#     --output "/projects/_ssd/ZhaoxuCode/Efficient-HA/Result/InternVL3-8B/output_greedy.json" \
#     --method "greedy" \
#     --max_tokens 512 \

# python generate_response_chair_llava.py \
#     --device "cuda:0" \
#     --model_id "OpenGVLab/InternVL3-8B" \
#     --output "/projects/_ssd/ZhaoxuCode/Efficient-HA/Result/InternVL3-8B/output_beam.json" \
#     --method "beam" \
#     --max_tokens 512 \

# python generate_response_chair_llava.py \
#     --device "cuda:0" \
#     --model_id "OpenGVLab/InternVL3-8B" \
#     --output "/projects/_ssd/ZhaoxuCode/Efficient-HA/Result/InternVL3-8B/output_dola.json" \
#     --method "dola" \
#     --max_tokens 512 \

# python generate_response_chair_llava.py \
#     --device "cuda:0" \
#     --model_id "OpenGVLab/InternVL3-8B" \
#     --output "/projects/_ssd/ZhaoxuCode/Efficient-HA/Result/InternVL3-8B/output_deco.json" \
#     --method "deco" \
#     --max_tokens 512 \


python generate_response_chair_llava.py \
    --device "cuda:0" \
    --model_id "OpenGVLab/InternVL3-8B" \
    --output "/projects/_ssd/ZhaoxuCode/Efficient-HA/Result/InternVL3-8B/output_ours.json" \
    --method "ours" \
    --max_tokens 512 \