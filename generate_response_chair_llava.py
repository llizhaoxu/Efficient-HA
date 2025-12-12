import gc
import os
import argparse
import json
import numpy as np
import torch
import re
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

from transformers import AutoProcessor, AutoModelForVision2Seq,CLIPImageProcessor,AutoModel,AutoTokenizer,AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
from model_utils.intervl_utils import load_image,chat
from model_utils.qianfan_utils import chat as qianfan_chat
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images
import time
from utils.vcd_add_noise import add_diffusion_noise,add_diffusion_noise_pil
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
from minigpt4.common.dist_utils import get_rank
from minigpt4.models import load_preprocess
from torchvision import transforms
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from PIL import Image, ImageOps
from utils.vcd_add_noise import add_diffusion_noise
from utils.vcd_sample import evolve_vcd_sampling
from utils.deco_greedy import evolve_deco_greedy
from utils.deco import generate as deco_generate
from utils.ours import generate as ours_generate
from utils.ours import ours as ours
import transformers
def jsd_batch(p, q, eps=1e-12):
    """
    p, q: [..., V] 概率分布张量
    返回: [...], 去掉最后一维 V
    """
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    m = 0.5 * (p + q)

    jsd_val = 0.5 * (
        torch.sum(p * (torch.log(p) - torch.log(m)), dim=-1) +
        torch.sum(q * (torch.log(q) - torch.log(m)), dim=-1)
    )
    return jsd_val   # shape [...], e.g. [1, T]
def load_model(model_id,args):
    """Load the model and processor."""
    if args.model_id == "Qwen/Qwen2.5-VL-3B-Instruct" or args.model_id == "Qwen/Qwen2.5-VL-7B-Instruct" or args.model_id == "VLM-Reasoner/LMM-R1-MGT-PerceReason" or args.model_id == "minglingfeng/Ocean_R1_3B_Instruct":

        min_pixels = 256 * 28 * 28
        max_pixels = 512 * 28 * 28
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True,
                                                min_pixels=min_pixels, max_pixels=max_pixels)

        model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        dtype='bfloat16',
        trust_remote_code=True,
        device_map=args.device
    )


        model.eval()
        return model, processor
    elif args.model_id == "OpenGVLab/InternVL3-8B":
        model = AutoModel.from_pretrained(
                model_id    ,
                torch_dtype=torch.bfloat16,
                load_in_8bit=False,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True,
                device_map=args.device).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)

        return model, tokenizer
    elif args.model_id == "microsoft/Phi-3.5-vision-instruct":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map=args.device, 
            trust_remote_code=True, 
            torch_dtype="auto", 
            )

            # for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
        processor = AutoProcessor.from_pretrained(model_id, 
            trust_remote_code=True, 
            num_crops=4
            ) 
        model.eval()
        return model, processor
    elif args.model_id == "baidu/Qianfan-VL-3B":
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=args.device
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        return model, tokenizer
    elif args.model_id == "deepseek-ai/deepseek-vl2-tiny":
        model = DeepseekVLV2ForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True
        ).to(torch.bfloat16).to(args.device)
        processor = DeepseekVLV2Processor.from_pretrained(model_id, trust_remote_code=True)
        return model, processor
    elif args.model_id == "ByteDance/Sa2VA-4B":
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True).eval().to(args.device)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        return model, tokenizer
    elif args.model_id == "llava-hf/llava-1.5-7b-hf":
        processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf",device_map="cuda:0")
        model = AutoModelForVision2Seq.from_pretrained("llava-hf/llava-1.5-7b-hf",device_map="cuda:0", dtype=torch.bfloat16)
        return model, processor
    elif args.model_id == "Salesforce/instructblip-vicuna-7b":
        model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b",dtype=torch.bfloat16,device_map="cuda:0",trust_remote_code=True)
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b",device_map="cuda:0")
        return model, processor

    else:
        raise ValueError(f"Model {args.model_id} not supported yet.")

def get_response(model, processor,args, image_path, question):

    if args.model_id == "Qwen/Qwen2.5-VL-3B-Instruct" or args.model_id == "Qwen/Qwen2.5-VL-7B-Instruct" or args.model_id == "VLM-Reasoner/LMM-R1-MGT-PerceReason" or args.model_id == "minglingfeng/Ocean_R1_3B_Instruct":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": (
                        question
                    )},
                ],
            }
        ]

        # Prepare inputs
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        lm_head = model.lm_head
        norm = model.language_model.norm

        if args.method == "ours":
            outputs = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False, use_ours=True,alpha=args.deco_alpha, threshold_top_p=args.deco_top_p, threshold_top_k=args.deco_top_k,
                                                early_exit_layers=[i for i in range(18,26)], lm_head=lm_head,
                    norm=norm,)
        elif args.method=="deco":
            outputs = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False, use_deco=True, alpha=args.deco_alpha, threshold_top_p=args.deco_top_p, threshold_top_k=args.deco_top_k,
                                                early_exit_layers=[i for i in range(20,29)], lm_head=lm_head,
                    norm=norm,)
        elif args.method=="dola":
            outputs = model.generate(**inputs, max_new_tokens=args.max_tokens,    custom_generate="transformers-community/dola",
    trust_remote_code=True,dola_layers='high', do_sample=False)
        elif args.method=="beam":
            outputs = model.generate(**inputs, max_new_tokens=args.max_tokens, num_beams=5, do_sample=False)
        elif args.method=="greedy":
            outputs = model.generate(**inputs, max_new_tokens=args.max_tokens,do_sample=False)
        generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
            ]
        output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

    elif args.model_id == "ByteDance/Sa2VA-4B":
        text_prompts = "<image>" + question
        image = Image.open(image_path).convert('RGB')
        input_dict = {
            'image': image,
            'text': text_prompts,
            'past_text': '',
            'mask_prompts': None,
            'tokenizer': tokenizer,
            }
        return_dict = model.predict_forward(**input_dict)
        output_text = return_dict["prediction"] #
    elif args.model_id == "microsoft/Phi-3.5-vision-instruct":
        images=[Image.open(image_path)]

        placeholder = "<|image_1|>\n"

        messages = [
            {"role": "user", "content": placeholder+question},
        ]

        prompt = processor.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
        )

        inputs = processor(prompt, images, return_tensors="pt").to(model.device)


        with torch.no_grad():
            if args.method == "greedy":
                generated_ids = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False)
            elif args.method == "beam":
                generated_ids = model.generate(**inputs, max_new_tokens=args.max_tokens, num_beams=5
                                            )
                
            elif args.method == "dola":
                generated_ids = model.generate(**inputs, max_new_tokens=args.max_tokens,dola_layers='high', do_sample=False)
            elif args.method == "deco":

                lm_head = model.lm_head
                norm = model.model.norm

                transformers.generation.utils.GenerationMixin.generate = deco_generate
                generated_ids = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False, alpha=args.deco_alpha, threshold_top_p=args.deco_top_p, threshold_top_k=args.deco_top_k,
                                            early_exit_layers=[i for i in range(16,24)], lm_head=lm_head,
                norm=norm)



            elif args.method == "vcd":
                evolve_vcd_sampling()

                inputs_cd= inputs.copy()
                inputs_cd['pixel_values'] = add_diffusion_noise(inputs['pixel_values'], args.noise_step)


                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    pixel_values_cd=(inputs_cd['pixel_values'].unsqueeze(0).half().cuda()),
                    cd_alpha=args.cd_alpha,
                    cd_beta=args.cd_beta,
                    do_sample=True)
            else:
                raise ValueError(f"Unknown generation method: {args.method}")
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

    elif args.model_id == "deepseek-ai/deepseek-vl2-tiny":
        tokenizer = processor.tokenizer
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image>\n{question}",
                "images": [image_path],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        pil_images = load_pil_images(conversation)
        prepare_inputs = processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        ).to(args.device)

        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)


        with torch.no_grad():
            if args.method == "greedy":
                generated_ids = model.generate(inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=args.max_tokens, do_sample=False)
            elif args.method == "beam":
                generated_ids = model.generate(    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,max_new_tokens=args.max_tokens, num_beams=5
                                            )
                
            elif args.method == "dola":
                generated_ids = model.generate(    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,max_new_tokens=args.max_tokens,dola_layers='high', do_sample=False,repetition_penalty=1.2)
            elif args.method == "deco":
                evolve_deco_greedy()

                generated_ids = model.generate(    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,max_new_tokens=args.max_tokens, top_p=None, top_k=None, do_sample=False, alpha=0.6, threshold_top_p=0.9, threshold_top_k=20,
                                            early_exit_layers=[i for i in range(22,32)], return_dict_in_generate=True,
                        output_hidden_states=True)
                
                generated_ids = generated_ids.sequences

            elif args.method == "vcd":
                evolve_vcd_sampling()

                inputs_cd= inputs.copy()
                inputs_cd['pixel_values'] = add_diffusion_noise(inputs['pixel_values'], args.noise_step)


                generated_ids = model.generate(
                        inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=args.max_tokens,
                    pixel_values_cd=(inputs_cd['pixel_values'].unsqueeze(0).half().cuda()),
                    cd_alpha=args.cd_alpha,
                    cd_beta=args.cd_beta,
                    do_sample=True)
            else:
                raise ValueError(f"Unknown generation method: {args.method}")

            output_text = tokenizer.decode(generated_ids[0].cpu().tolist(), skip_special_tokens=True)
       
    elif args.model_id == "baidu/Qianfan-VL-3B":
        pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).to(args.device)

        with torch.no_grad():
            if args.method == "greedy":
                generation_config = dict(max_new_tokens=args.max_tokens, do_sample=False)

            elif args.method == "beam":
                generation_config = dict(max_new_tokens=args.max_tokens, num_beams=5, do_sample=False)

                
            elif args.method == "dola":
                generation_config = dict(max_new_tokens=args.max_tokens,dola_layers='high', do_sample=False,repetition_penalty=1.2)

            elif args.method == "deco":

                lm_head = model.language_model.lm_head
                norm = model.language_model.model.norm
                transformers.generation.utils.GenerationMixin.generate = deco_generate
                generation_config =  dict(max_new_tokens=args.max_tokens, do_sample=False, alpha=args.deco_alpha, threshold_top_p=args.deco_top_p, threshold_top_k=args.deco_top_k,
                                            early_exit_layers=[i for i in range(18, 27)], lm_head=lm_head,
                norm=norm,)

            elif args.method == "vcd":
                evolve_vcd_sampling()
                
                inputs_cd= inputs.copy()
                inputs_cd['pixel_values'] = add_diffusion_noise(inputs['pixel_values'], args.noise_step)
                generation_config = dict(max_new_tokens=args.max_tokens, pixel_values_cd=(inputs_cd['pixel_values'].unsqueeze(0).half().cuda()), cd_alpha=args.cd_alpha, cd_beta=args.cd_beta, do_sample=True)
            question='<image>'+question
            output_text= qianfan_chat(model, processor, pixel_values, question, generation_config)
    elif args.model_id == "OpenGVLab/InternVL3-8B":
        pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).to(args.device)

        with torch.no_grad():
            if args.method == "greedy":
                generation_config = dict(max_new_tokens=args.max_tokens, do_sample=False)

            elif args.method == "beam":
                generation_config = dict(max_new_tokens=args.max_tokens, num_beams=5, do_sample=False)

                
            elif args.method == "dola":
                generation_config = dict(max_new_tokens=args.max_tokens,custom_generate="transformers-community/dola",dola_layers='high', do_sample=False,repetition_penalty=1.2)

            elif args.method == "deco": 

                lm_head = model.language_model.lm_head
                norm = model.language_model.model.norm
                
                generation_config = dict(max_new_tokens=args.max_tokens, do_sample=False, use_deco=True,alpha=args.deco_alpha, threshold_top_p=args.deco_top_p, threshold_top_k=args.deco_top_k,
                                                early_exit_layers=[i for i in range(20,29)], lm_head=lm_head,
                    norm=norm,)


            elif args.method == "ours":
                lm_head = model.language_model.lm_head
                norm = model.language_model.model.norm
                
                generation_config = dict(max_new_tokens=args.max_tokens, do_sample=False, use_ours=True,alpha=args.deco_alpha, threshold_top_p=args.deco_top_p, threshold_top_k=args.deco_top_k,
                                                early_exit_layers=[i for i in range(18,26)], lm_head=lm_head,
                    norm=norm,)
            question='<image>\n'+question
            output_text= chat(model, processor, pixel_values, question, generation_config)

    elif args.model_id == "llava-hf/llava-1.5-7b-hf":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": (
                        question
                    )},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        lm_head = model.lm_head
        norm = model.language_model.norm
        if args.method == "ours":
            outputs = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False, use_ours=True,alpha=args.deco_alpha, threshold_top_p=args.deco_top_p, threshold_top_k=args.deco_top_k,
                                                early_exit_layers=[i for i in range(18,26)], lm_head=lm_head,
                    norm=norm,)
        elif args.method=="deco":
            outputs = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False, use_deco=True, alpha=args.deco_alpha, threshold_top_p=args.deco_top_p, threshold_top_k=args.deco_top_k,
                                                early_exit_layers=[i for i in range(20,29)], lm_head=lm_head,
                    norm=norm,)
        elif args.method=="dola":
            outputs = model.generate(**inputs, max_new_tokens=args.max_tokens,    custom_generate="transformers-community/dola",
    trust_remote_code=True,dola_layers='high', do_sample=False)
        elif args.method=="beam":
            outputs = model.generate(**inputs, max_new_tokens=args.max_tokens, num_beams=5, do_sample=False)
        elif args.method=="greedy":
            outputs = model.generate(**inputs, max_new_tokens=args.max_tokens,do_sample=False)
        generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
            ]
        output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
 
        return output_text
    elif args.model_id == "Salesforce/instructblip-vicuna-7b":
        image = Image.open(image_path).convert("RGB")
        prompt = question
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

        lm_head = model.language_model.lm_head
        norm = model.language_model.model.norm
        if args.method == "ours":
            outputs = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False, use_ours=True,alpha=args.deco_alpha, threshold_top_p=args.deco_top_p, threshold_top_k=args.deco_top_k,
                                                early_exit_layers=[i for i in range(18,26)], lm_head=lm_head,
                    norm=norm,)
        elif args.method=="deco":
            outputs = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False, use_deco=True, alpha=args.deco_alpha, threshold_top_p=args.deco_top_p, threshold_top_k=args.deco_top_k,
                                                early_exit_layers=[i for i in range(20,29)], lm_head=lm_head,
                    norm=norm,)
        elif args.method=="dola":
            outputs = model.generate(**inputs, max_new_tokens=args.max_tokens,    custom_generate="transformers-community/dola",
    trust_remote_code=True,dola_layers='high', do_sample=False)
        elif args.method=="beam":
            outputs = model.generate(**inputs, max_new_tokens=args.max_tokens, num_beams=5, do_sample=False)
        elif args.method=="greedy":
            outputs = model.generate(**inputs, max_new_tokens=args.max_tokens,do_sample=False)
        
        generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
            ]
        output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        return output_text

    return output_text

def process_json(model, processor, args, output_json):
    image_ids = []
    with open('/projects/_ssd/ZhaoxuCode/Efficient-HA/opera_log/llava-1.5/greedy.jsonl', "r", encoding="utf-8") as f:
            for line in f.readlines():
                json_data = json.loads(line)
                json_data.pop('caption')
                image_ids.append(json_data)


    total_samples = len(image_ids)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    if not os.path.exists(output_json):
        with open(output_json, 'w') as f:
            json.dump([], f)
    if not os.path.exists(args.output_hidden):
        with open(args.output_hidden, 'w') as f:
            json.dump([], f)
    with open(output_json, 'r') as f:

        current_data=json.load(f) 
    processed_idx=[item['image_id'] for item in current_data]


    question ='Describe this image in detail.'
    
    error_id=[]
    i=0
    for idx, line in enumerate(image_ids):
        if idx in processed_idx:
            continue
        image_path = "COCO_val2014_" + str(line['image_id']).zfill(12) + ".jpg"
        image_path = os.path.join(args.datapath, image_path)


        if args.model_id == "llava-hf/llava-1.5-7b-hf":
            response = get_response(model, processor,args, image_path, question)


            torch.cuda.empty_cache()

            line['response'] = response


            print(f"Processed sample {idx + 1}/{total_samples}")

            with open(output_json, 'r') as f:
                current_data = json.load(f)

            current_data.append(line)

            with open(output_json, 'w') as f:
                json.dump(current_data, f, indent=2)
            
            
        else:

            response = get_response(model, processor,args, image_path, question)
    

            torch.cuda.empty_cache()

            line['response'] = response


            print(f"Processed sample {idx + 1}/{total_samples}")

            with open(output_json, 'r') as f:
                current_data = json.load(f)

            current_data.append(line)

            with open(output_json, 'w') as f:
                json.dump(current_data, f, indent=2)

    print(error_id)

    print(f"All results saved to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str,
                        default='/projects/_ssd/ZhaoxuCode/Efficient-HA/hidden_1/output_greedy.json',
                        help='Output file to store model responses')
    parser.add_argument('--output_hidden', type=str,
                        default='/projects/_ssd/ZhaoxuCode/Efficient-HA/hidden_1/hidden_greedy.json',
                        help='Output file to store hidden states')
    parser.add_argument('--model_id', type=str, default="OpenGVLab/InternVL3-8B",
                        help='Path to the model')
    
    parser.add_argument('--datapath', type=str, default="/projects/_hdd/Datazx/coco_val_images/val2014",
                        help='Path to the data')
    parser.add_argument('--method', type=str, default="ours")
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--deco_alpha", type=float, default=0.6)
    parser.add_argument("--deco_top_p", type=float, default=0.9)
    parser.add_argument("--deco_top_k", type=int, default=20)

    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--max_tokens', type=int, default=512)
    args = parser.parse_args()

    model, processor = load_model(args.model_id,args)

    process_json(model, processor,args, args.output)

