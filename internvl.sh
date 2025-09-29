

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
###mme
python generate_response_mme.py \
    --device "cuda:4" \
    --model_id "OpenGVLab/InternVL3-2B" \
    --output "result/mme/InternVL3-2B/output_greedy.json" \
    --method "greedy" \
    --max_tokens 64 \

python generate_response_mme.py \
    --device "cuda:4" \
    --model_id "OpenGVLab/InternVL3-2B" \
    --output "result/mme/InternVL3-2B/output_beam.json" \
    --method "beam" \
    --max_tokens 64 \

python generate_response_mme.py \
    --device "cuda:4" \
    --model_id "OpenGVLab/InternVL3-2B" \
    --output "result/mme/InternVL3-2B/output_dola.json" \
    --method "dola" \
    --max_tokens 64 \


python generate_response_mme.py \
    --device "cuda:4" \
    --model_id "OpenGVLab/InternVL3-2B" \
    --output "result/mme/InternVL3-2B/output_deco.json" \
    --method "deco" \
    --max_tokens 64 \


###pope


python generate_response_pope.py \
    --device "cuda:4" \
    --model_id "OpenGVLab/InternVL3-2B" \
    --output "result/pope/InternVL3-2B" \
    --pope_type "random" \
    --method "greedy" \
    --max_tokens 10 \

python generate_response_pope.py \
    --device "cuda:4" \
    --model_id "OpenGVLab/InternVL3-2B" \
    --output "result/pope/InternVL3-2B" \
    --pope_type "random" \
    --method "beam" \
    --max_tokens 10 \

python generate_response_pope.py \
    --device "cuda:4" \
    --model_id "OpenGVLab/InternVL3-2B" \
    --output "result/pope/InternVL3-2B" \
    --pope_type "random" \
    --method "dola" \
    --max_tokens 10 \


python generate_response_pope.py \
    --device "cuda:4" \
    --model_id "OpenGVLab/InternVL3-2B" \
    --output "result/pope/InternVL3-2B" \
    --pope_type "random" \
    --method "deco" \
    --max_tokens 10 \


######
python generate_response_pope.py \
    --device "cuda:4" \
    --model_id "OpenGVLab/InternVL3-2B" \
    --output "result/pope/InternVL3-2B" \
    --pope_type "popular" \
    --method "greedy" \
    --max_tokens 10 \

python generate_response_pope.py \
    --device "cuda:4" \
    --model_id "OpenGVLab/InternVL3-2B" \
    --output "result/pope/InternVL3-2B" \
    --pope_type "popular" \
    --method "beam" \
    --max_tokens 10 \

python generate_response_pope.py \
    --device "cuda:4" \
    --model_id "OpenGVLab/InternVL3-2B" \
    --output "result/pope/InternVL3-2B" \
    --pope_type "popular" \
    --method "dola" \
    --max_tokens 10 \


python generate_response_pope.py \
    --device "cuda:4" \
    --model_id "OpenGVLab/InternVL3-2B" \
    --output "result/pope/InternVL3-2B" \
    --pope_type "popular" \
    --method "deco" \
    --max_tokens 10 \


#######
python generate_response_pope.py \
    --device "cuda:4" \
    --model_id "OpenGVLab/InternVL3-2B" \
    --output "result/pope/InternVL3-2B" \
    --pope_type "adversarial" \
    --method "greedy" \
    --max_tokens 10 \

python generate_response_pope.py \
    --device "cuda:4" \
    --model_id "OpenGVLab/InternVL3-2B" \
    --output "result/pope/InternVL3-2B" \
    --pope_type "adversarial" \
    --method "beam" \
    --max_tokens 10 \

python generate_response_pope.py \
    --device "cuda:4" \
    --model_id "OpenGVLab/InternVL3-2B" \
    --output "result/pope/InternVL3-2B" \
    --pope_type "adversarial" \
    --method "dola" \
    --max_tokens 10 \


python generate_response_pope.py \
    --device "cuda:4" \
    --model_id "OpenGVLab/InternVL3-2B" \
    --output "result/pope/InternVL3-2B" \
    --pope_type "adversarial" \
    --method "deco" \
    --max_tokens 10 \

###amber

python generate_response_AMBER.py \
    --device "cuda:4" \
    --model_id "OpenGVLab/InternVL3-2B" \
    --output "result/AMBER/InternVL3-2B/output_greedy.json" \
    --method "greedy" \
    --max_tokens 64 \

python generate_response_AMBER.py \
    --device "cuda:4" \
    --model_id "OpenGVLab/InternVL3-2B" \
    --output "result/AMBER/InternVL3-2B/output_beam.json" \
    --method "beam" \
    --max_tokens 64 \

python generate_response_AMBER.py \
    --device "cuda:4" \
    --model_id "OpenGVLab/InternVL3-2B" \
    --output "result/AMBER/InternVL3-2B/output_dola.json" \
    --method "dola" \
    --max_tokens 64 \

python generate_response_AMBER.py \
    --device "cuda:4" \
    --model_id "OpenGVLab/InternVL3-2B" \
    --output "result/AMBER/InternVL3-2B/output_deco.json" \
    --method "deco" \
    --max_tokens 64 \