

python generate_response_chair.py \
    --device "cuda:4" \
    --model_id "OpenGVLab/InternVL3-2B" \
    --output "result/chair/InternVL3-2B/output_greedy.json" \
    --method "greedy" \
    --max_tokens 64 \

python generate_response_chair.py \
    --device "cuda:4" \
    --model_id "OpenGVLab/InternVL3-2B" \
    --output "result/chair/InternVL3-2B/output_beam.json" \
    --method "beam" \
    --max_tokens 64 \

python generate_response_chair.py \
    --device "cuda:4" \
    --model_id "OpenGVLab/InternVL3-2B" \
    --output "result/chair/InternVL3-2B/output_dola.json" \
    --method "dola" \
    --max_tokens 64 \


python generate_response_chair.py \
    --device "cuda:4" \
    --model_id "OpenGVLab/InternVL3-2B" \
    --output "result/chair/InternVL3-2B/output_deco.json" \
    --method "deco" \
    --max_tokens 64 \

python generate_response_mme.py \
    --device "cuda:6" \
    --model_id "Qwen/Qwen2.5-VL-7B-Instruct" \
    --output "result/mme/Qwen2.5-VL-7B-Instruct/output_greedy.json" \
    --method "greedy" \
    --max_tokens 64 \