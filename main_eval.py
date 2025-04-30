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
import random 
from dataset import TwoDigitArithmeticDataset

from pdb import set_trace as st 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument("--head_str", type=str)

    parser.add_argument("--arithmetic_type", type=str, default="multiplication")
    parser.add_argument("--layer_id", type=int, default=-1)
    parser.add_argument('--attn_head_list', type=int, default=-1, nargs='+', help='List of integers separated by space')

    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--access_token", type=str, default="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")
    parser.add_argument("--savedir", type=str, default=".")
    args = parser.parse_args()


    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    random.seed(args.seed)
    model, tokenizer = lib.load_llm_hf(args)
    device = torch.device("cuda:0")
    tokenizer.pad_token_id = tokenizer.eos_token_id  ## set up for batch inference 

    dataset = TwoDigitArithmeticDataset(arithmetic_type=args.arithmetic_type)

    layers = model.model.layers

    ### set the layer and the attention heads to zero out;
    if args.layer_id != -1:
        layer_id = args.layer_id
        if args.attn_head_list == -1:
            attn_id_list = random.sample(range(32), 5)
        else:
            attn_id_list = args.attn_head_list
        mp.zero_ablate_llama3_attn_head(layers[layer_id], attn_id_list)

    print("length of dataset ", len(dataset))

    if args.head_str is None:
        f=open(os.path.join(args.savedir, "default.txt"), "w")
    else:
        f=open(os.path.join(args.savedir, f"{args.head_str}.txt"), "w")

    cor = 0

    sampled_indices = random.sample(range(len(dataset)), k=100)

    message_list = []
    input_strs = []
    label_strs = []
    for i in sampled_indices:
        input_str = dataset[i]["input"]
        label = dataset[i]["target"]
        message_list.append([{"role": "user", "content": "Answer the question directly:\n"+input_str}])
        input_strs.append(input_str)
        label_strs.append(label)
    texts = tokenizer.apply_chat_template(message_list, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(texts, padding="longest", padding_side="left", return_tensors="pt")
    inputs = {key: val.to(model.device) for key, val in inputs.items()}
    input_ids = inputs["input_ids"]
    # st()

    outputs = model.generate(**inputs, do_sample=False, max_new_tokens=50)
    decoded_strs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    ### post-process the decoded string;
    decoded_strs = [decoded_str.split("assistant\n\n")[1] for decoded_str in decoded_strs]
    decoded_strs = [decoded_str.strip(".").replace(",","") for decoded_str in decoded_strs]

    cor = 0 
    for input_str, decoded_str, label_str in zip(input_strs, decoded_strs, label_strs):
        print(f"{input_str}\t\t{decoded_str}", file=f,flush=True)
        cor += (decoded_str == label_str)

    acc = cor / len(sampled_indices)
    print(f"accuracy is {acc}", file=f, flush=True)

if __name__ == '__main__':
    main()