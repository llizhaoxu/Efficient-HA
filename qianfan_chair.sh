

python generate_response_chair.py \
    --device "cuda:6" \
    --model_id "deepseek-ai/deepseek-vl2-tiny" \
    --output "result/chair/Qianfan-VL-3B/output_greedy.json" \
    --method "greedy" \
    --max_tokens 64 \

python generate_response_chair.py \
    --device "cuda:6" \
    --model_id "deepseek-ai/deepseek-vl2-tiny" \
    --output "result/chair/Qianfan-VL-3B/output_beam.json" \
    --method "beam" \
    --max_tokens 64 \

python generate_response_chair.py \
    --device "cuda:6" \
    --model_id "deepseek-ai/deepseek-vl2-tiny" \
    --output "result/chair/Qianfan-VL-3B/output_dola.json" \
    --method "dola" \
    --max_tokens 64 \



python generate_response_chair.py \
    --device "cuda:6" \
    --model_id "deepseek-ai/deepseek-vl2-tiny" \
    --output "result/chair/Qianfan-VL-3B/output_deco.json" \
    --method "deco" \
    --max_tokens 64 \