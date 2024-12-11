import torch 
import argparse
import os 
import torch.nn as nn 

import numpy as np 
import monkey_patch as mp
from lib.model_dict import MODEL_DICT_LLMs
import lib 
import torch.nn.functional as F
import numpy as np
import re 
from datasets import load_dataset
from tqdm import tqdm

from pdb import set_trace as st 

def extract_head_info(head_info):
    match = re.match(r'L(\d)H(\d)', head_info)
    if match:
        numbers = [int(match.group(1)), int(match.group(2))]
        return numbers
    else:
        raise ValueError("Invalid head_info")

def get_loss(model, tokenizer, prompt, device = torch.device("cuda:0")):
    input_ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(input_ids)

    cross_entropy_loss = []
    logits = outputs.logits 
    logits = logits.view(-1, logits.shape[-1])[:-1,:]
    labels = input_ids.view(-1)[1:]
    loss = F.cross_entropy(logits, labels, reduction="none")
    return loss 

def collect_mean_ablation(args, model, tokenizer, device = torch.device("cuda:0")):
    ds = load_dataset("lmsys/lmsys-chat-1m", cache_dir="/data/locus/project_data/project_data3/mingjies/huggingface")

    layers = model.model.layers
    ### collect mean head stats
    for layer_id in range(len(layers)):
        mp.mean_ablate_llama3_attn_head(layers[layer_id], 0, collect_stats=True)

    #### code for zero ablation 
    for i in tqdm(range(args.num_samples)):
        prompt = ds["train"][i]["conversation"][0]["content"]
        get_loss(model, tokenizer, prompt)

    outputs = []
    for layer_id in range(len(layers)):
        mean_attn_outputs = layers[layer_id].self_attn.mean_head_stats
        outputs.append(mean_attn_outputs)
    torch.save(outputs, os.path.join(args.savedir, f"mean_outputs_{args.num_samples}.pt"))

def analyze():
    obj_default = torch.load("results/llama3.1-8b-it/mean_ablate/token_agnostic/no_ablate.pt")
    obj_mean = torch.load("results/llama3.1-8b-it/mean_ablate/token_agnostic/mean_ablate_L0H0.pt")

    # for i in range(len(obj_default)):
    #     pass 
    exit()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument("--head_str", type=str, default="L0H0")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--collect_mean", action="store_true")
    parser.add_argument("--access_token", type=str, default="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")
    parser.add_argument("--savedir", type=str, default=".")
    args = parser.parse_args()

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    model, tokenizer = lib.load_llm_hf(args)
    ### This code add a hook to the model 
    layers = model.model.layers

    layer_id, attn_id = extract_head_info(args.head_str)

    if args.collect_mean:
        collect_mean_ablation(args, model, tokenizer)
        return
    # analyze()

    # prompt = "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    ds = load_dataset("lmsys/lmsys-chat-1m", cache_dir="/data/locus/project_data/project_data3/mingjies/huggingface")

    #### mean ablate llama3 attn_head;
    obj = torch.load("results/llama3.1-8b-it/mean_ablate/token_agnostic/mean_outputs_100.pt")
    layers[layer_id].self_attn.mean_head_stats = obj[layer_id]
    mp.mean_ablate_llama3_attn_head(layers[layer_id], attn_id, collect_stats=False)

    obj = []
    #### collecting the mean ablation token-wise loss 
    for i in tqdm(range(args.num_samples)):
        prompt = ds["train"][i]["conversation"][0]["content"]
        loss = get_loss(model, tokenizer, prompt)
        obj.append({"prompt": prompt, "loss": loss.cpu()})
    torch.save(obj, os.path.join(args.savedir, f"mean_ablate_{args.head_str}.pt"))

if __name__ == '__main__':
    main()