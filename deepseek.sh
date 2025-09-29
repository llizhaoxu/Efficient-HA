

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