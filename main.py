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

# def collect_mean_ablation(model, tokenizer, prompt, device = torch.device("cuda:0")):
#     pass 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument("--head_str", type=str, default="L0H0")
    parser.add_argument("--access_token", type=str, default="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")
    parser.add_argument("--savedir", type=str, default=".")
    args = parser.parse_args()

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    model, tokenizer = lib.load_llm_hf(args)
    ### This code add a hook to the model 
    layers = model.model.layers

    layer_id, attn_id = extract_head_info(args.head_str)

    # prompt = "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    ds = load_dataset("lmsys/lmsys-chat-1m", cache_dir="/data/locus/project_data/project_data3/mingjies/huggingface")

    #### code for zero ablation 
    for i in range(5):
        prompt = ds["train"][i]["conversation"][0]["content"]
        loss1 = get_loss(model, tokenizer, prompt)
        mp.zero_ablate_llama3_attn_head(layers[layer_id], attn_id)
        loss2 = get_loss(model, tokenizer, prompt)

        lossdiff = loss2 - loss1   ## we want those where the loss increases after ablation;

if __name__ == '__main__':
    main()