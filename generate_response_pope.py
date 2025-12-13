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
from PIL import Image, ImageOps
from tqdm import tqdm
from utils.vcd_add_noise import add_diffusion_noise
from utils.vcd_sample import evolve_vcd_sampling
from utils.deco_greedy import evolve_deco_greedy
from utils.deco import generate as deco_generate
from utils.ours import generate as ours_generate
from utils.ours import ours as ours
from pope_loader import POPEDataSet
import transformers
POPE_PATH = {
    "random": "pope_coco/coco_pope_random.json",
    "popular": "pope_coco/coco_pope_popular.json",
    "adversarial": "pope_coco/coco_pope_adversarial.json",
}
def print_acc(pred_list, label_list,args,base_dir):
    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)
    # unknown_ratio = pred_list.count(2) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))
    os.makedirs(base_dir, exist_ok=True)
    with open(os.path.join(base_dir, 'POPE_type_{}_{}.jsonl'.format(args.pope_type,args.method)), "a") as f:
        json.dump({
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Yes_ratio": yes_ratio,
        }, f)
        f.write('\n')
    
        

def recorder(out, pred_list):
    NEG_WORDS = ["No", "not", "no", "NO"]
    line=out

    line = line.replace('.', '')
    line = line.replace(',', '')
    words = line.split(' ')
    if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
        pred_list.append(0)
    else:
        pred_list.append(1)
    
    return pred_list

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


def process_json(model, processor, args, output):
    args.pope_path = POPE_PATH[args.pope_type]
    pope_dataset = POPEDataSet(
        pope_path=args.pope_path, 
        data_path=args.datapath, 
    )
    pope_loader = torch.utils.data.DataLoader(
        pope_dataset, 
        batch_size=1, 
        shuffle=False, 
        drop_last=False
    )

    total_samples = len(pope_dataset)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    caption_file = os.path.join(output, "POPE_caption_{}_{}.json".format(args.pope_type,args.method))
    os.makedirs(os.path.dirname(caption_file), exist_ok=True)
    if not os.path.exists(caption_file):
        with open(caption_file, 'w') as f:
            json.dump([], f)
    pred_list, pred_list_s, label_list = [], [], []

    for batch_id, data in tqdm(enumerate(pope_loader), total=len(pope_loader)):
 
        image_path = data["image_path"][0]
        qu = data["query"][0]
        label = data["label"]
        label_list = label_list + list(label)

        label = torch.Tensor(label).to(model.device)
        
        response = get_response(model, processor,args, image_path, qu)

        torch.cuda.empty_cache()
        pred_list = recorder(response, pred_list)
        with open(caption_file, 'r') as f:
            current_data = json.load(f)
        current_data.append({
            "image_path": image_path,
            "query": qu,
            "response": response,
            "label": label[0].item(),
        })
        with open(caption_file, 'w') as f:
            json.dump(current_data, f,indent=4)


    if len(pred_list) != 0:
        print_acc(pred_list, label_list,args,args.output)
    if len(pred_list_s) != 0:
        print_acc(pred_list_s, label_list,args,args.output)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str,
                        default='/projects/_ssd/ZhaoxuCode/Efficient-HA/hidden_10/test',
                        help='Output file to store model responses')

    parser.add_argument("--pope_type", type=str, help="random",default='random', choices=['random', 'popular', 'adversarial'])
    parser.add_argument('--model_id', type=str, default="llava-hf/llava-1.5-7b-hf",
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
    parser.add_argument('--max_tokens', type=int, default=10)
    args = parser.parse_args()

    model, processor = load_model(args.model_id,args)

    process_json(model, processor,args, args.output)

