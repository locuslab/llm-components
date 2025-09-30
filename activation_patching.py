import torch 
import argparse
import os 
import random 
import torch.nn as nn 
import numpy as np 

from dataset import TwoDigitArithmeticDataset, GSM8k
from modify_model import activation_patching_attn_head
from load_model import load_llm_hf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    # parser.add_argument("--head_str", type=str)

    parser.add_argument("--arithmetic_type", type=str, default="multiplication")
    parser.add_argument("--layer_id", type=int, default=-1)
    parser.add_argument('--attn_head_list', type=int, default=-1, nargs='+', help='List of integers separated by space')

    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--access_token", type=str)
    parser.add_argument("--savedir", type=str, default=".")

    parser.add_argument("--collect_pre_act", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    random.seed(args.seed)
    model, tokenizer = load_llm_hf(args)
    device = torch.device("cuda:0")
    tokenizer.pad_token_id = tokenizer.eos_token_id  ## set up for batch inference 

    layers = model.model.layers

    # if args.layer_id != -1:
    #     layer_id = args.layer_id
    #     if args.attn_head_list == -1:
    #         attn_id_list = random.sample(range(32), 5)
    #     else:
    attn_id_list = args.attn_head_list
    activation_patching_attn_head(layers[args.layer_id], attn_id_list, args.collect_pre_act, args)
    # for layer_id, layer in enumerate(layers):
    #     activation_patching_attn_head(layer, attn_id_list, args.collect_pre_act, layer_id, args)

    dataset = TwoDigitArithmeticDataset(arithmetic_type=args.arithmetic_type, modeltype=args.model)
    print("length of dataset ", len(dataset))

    sampled_indices = random.sample(range(len(dataset)), k=100)
    # import pdb; pdb.set_trace()

    message_list = []
    input_strs = []
    label_strs = []
    for i in sampled_indices:
        if args.collect_pre_act:
            # get activations if we were doing the equivalent subtraction problem instead of addition
            input_str = dataset[i]["input"].replace('x', '+')
            # input_str = "32 - 1 ="
        else:
            input_str = dataset[i]["input"]
        label = dataset[i]["target"]
        message_list.append([{"role": "user", "content": "Answer the question directly:\n"+input_str}])
        input_strs.append(input_str)
        label_strs.append(label)
    print('message: ', message_list)
    texts = tokenizer.apply_chat_template(message_list, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(texts, padding="longest", padding_side="left", return_tensors="pt")
    inputs = {key: val.to(model.device) for key, val in inputs.items()}
    input_ids = inputs["input_ids"]

    # this is where the forward pass is happening and the saving will occur
    outputs = model.generate(**inputs, do_sample=False, max_new_tokens=50)
    decoded_strs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    decoded_strs = [decoded_str.split("assistant\n\n")[1] for decoded_str in decoded_strs]
    decoded_strs = [decoded_str.strip(".").replace(",","") for decoded_str in decoded_strs]
    cor = 0 
    for input_str, decoded_str, label_str in zip(input_strs, decoded_strs, label_strs):
        print(f"{input_str}\t{decoded_str} | {label_str}")
        cor += (decoded_str == label_str)

    acc = cor / len(sampled_indices)
    print(f"accuracy is {acc}")

if __name__ == '__main__':
    main()
