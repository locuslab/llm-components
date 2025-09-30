import torch 
import argparse
import os 
import random 
import numpy as np 
from tqdm import tqdm
import ast

import torch.nn as nn 

from modify_model import zero_ablate_attn_head, mean_ablate_attn_head
from load_model import load_llm_hf
import dataset as ds

def clean_print(token_list):
    new_list = []
    for word in token_list:
        if word[0] == 'Ġ':
            new_list.append(word[1:])
        else:
            new_list.append(word)
    return new_list


def get_name(args):
    shorthands = ['llama3.1', 'llama3.2', 'qwen']
    for name in shorthands:
        if name in args.model:
            if '3b' in args.model:
                return name + '_3b'
            elif '1b' in args.model:
                return name + '_1b'
            elif '7b' in args.model:
                return name + '_7b'
            elif '14b' in args.model:
                return name + '_14b'
            elif '13b' in args.model:
                return name + '_13b'
            else:
                return name


def tokenizer_dict(tokenizer):
    return {str(tokenizer(str(num))['input_ids'][1]): [tokenizer(ch)['input_ids'][1] for ch in list(str(num))] for num in range(10000)}


def modified_tokenizer(tokenizer, texts):
    pad = 128001
    dct = tokenizer_dict(tokenizer)
    inputs = tokenizer(texts, padding="longest", padding_side="left", return_tensors="pt")
    retokenized_inputs = [] 
    for row in inputs['input_ids']:
        new_row = []
        for i in row:
            if str(i.item()) in dct:
                new_row.extend(dct[str(i.item())])
            else:
                new_row.append(i.item())
        retokenized_inputs.append(new_row)
    max_len = max([len(row) for row in retokenized_inputs])
    new_input_ids = []
    new_attention_masks =[]
    for row in retokenized_inputs:
        rowlen = len(row)
        if rowlen < max_len:
            extra = max_len - rowlen
            row = [pad] * extra + row
            attn_row = [0] * extra + [1] * rowlen
        else:
            attn_row = [1] * rowlen
        new_input_ids.append(row)
        new_attention_masks.append(attn_row)
    new_inputs = {'input_ids': torch.tensor(new_input_ids), 'attention_mask': torch.tensor(new_attention_masks)}
    return new_inputs

def collect_mean_ablation(args, model, num_heads):
    """
    Collect mean head statistics across the whole model
    Saves mean values in mean_outputs_{model}_{num_samples}.pt
    """
    layers = model.model.layers
    for layer_id in range(len(layers)):
        for head_id in range(num_heads):
            mean_ablate_attn_head(layers[layer_id], [head_id], num_heads, collect_stats=True)
    outputs = []
    for layer_id in range(len(layers)):
        mean_attn_outputs = layers[layer_id].self_attn.mean_head_stats
        outputs.append(mean_attn_outputs)
    torch.save(outputs, os.path.join(args.savedir, f"mean_outputs_{args.model}_{args.num_samples}.pt"))

def extract_head_info(head_info):
    match = re.match(r'L(\d{1,2})H(\d{1,2})', head_info)
    if match:
        numbers = [int(match.group(1)), int(match.group(2))]
        return numbers
    else:
        raise ValueError("Invalid head_info")


def main(args):
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    random.seed(args.seed)

    model, tokenizer = load_llm_hf(args)
    model.config.use_cache = False
    device = torch.device("cuda:0")
    tokenizer.pad_token_id = tokenizer.eos_token_id 
    layers = model.model.layers
    
    # Keeps only the heads in attn_head_list out of the entire model
    # this didn't really work so we probably won't use it
    if args.single_head:
        if args.mean_ablation:
            num_heads = model.config.num_attention_heads
            collect_mean_ablation(args, model, num_heads)
            layers = model.model.layers
            obj = torch.load(os.path.join(args.savedir, f"mean_outputs_{args.model}_{args.num_samples}.pt"), weights_only=True)
            layers[args.layer_id].self_attn.mean_head_stats = obj[args.layer_id]
            mean_ablate_attn_head(layers[args.layer_id], args.attn_head_list, num_heads, collect_stats=False)
        else:
            num_heads = model.config.num_attention_heads
            knockout = [i for i in range(num_heads) if i not in args.attn_head_list]
            all_heads = [i for i in range(num_heads)]
            for layer_idx, layer in enumerate(layers):
                if layer_idx == args.layer_id:
                    zero_ablate_attn_head(layer, knockout, args.model)
                else: 
                    zero_ablate_attn_head(layer, all_heads, args.model)
    # Perform ablation of head(s) if a layer(s) is specified
    elif args.layer_id is not None or args.layer_id_list != -1:
        if args.complement:
            num_heads = model.config.num_attention_heads
            knockout = [i for i in range(num_heads) if i not in args.attn_head_list]
            if args.mean_ablation:
                # collect_mean_ablation(args, model, num_heads)
                obj = torch.load(os.path.join(args.savedir, f"mean_outputs_{args.model}_{args.num_samples}.pt"), weights_only=True)
                layers[args.layer_id].self_attn.mean_head_stats = obj[args.layer_id]
                mean_ablate_attn_head(layers[args.layer_id], knockout, num_heads, collect_stats=True)
            else:
                zero_ablate_attn_head(layers[args.layer_id], knockout, args.model)

        else:
            num_heads = model.config.num_attention_heads
            # # collect_mean_ablation(args, model, num_heads)
            if args.mean_ablation:
                if args.layer_id_list != -1: 
                    for pair_idx in range(len(args.layer_id_list)):
                        layer_idx = args.layer_id_list[pair_idx]
                        head_idx = args.attn_head_list[pair_idx]
                        obj = torch.load(os.path.join(args.savedir, f"mean_outputs_{args.model}_{args.num_samples}.pt"), weights_only=True)
                        layers[layer_idx].self_attn.mean_head_stats = obj[layer_idx]
                        mean_ablate_attn_head(layers[layer_idx], [head_idx], num_heads, collect_stats=True)
                else:
                    layers = model.model.layers
                    obj = torch.load(os.path.join(args.savedir, f"mean_outputs_{args.model}_{args.num_samples}.pt"), weights_only=True)
                    layers[args.layer_id].self_attn.mean_head_stats = obj[args.layer_id]
                    mean_ablate_attn_head(layers[args.layer_id], args.attn_head_list, num_heads, collect_stats=True)                
            else:
                if args.layer_id_list != -1: 
                    for pair_idx in range(len(args.layer_id_list)):
                        layer_idx = args.layer_id_list[pair_idx]
                        head_idx = args.attn_head_list[pair_idx]
                        zero_ablate_attn_head(layers[layer_idx], [head_idx])
                else:
                    zero_ablate_attn_head(layers[args.layer_id], args.attn_head_list) 

    # Obtain data
    if args.dataset == 'twodig':
        dataset = ds.TwoDigitArithmeticDataset(arithmetic_type=args.arithmetic_type, modeltype=args.model)
    elif args.dataset == 'rhyme':
        dataset = ds.EasyRhymeDataset()
    elif args.dataset == 'arithmetic_coding':
        dataset = ds.SimpleArithmeticProgrammingDataset(n_per_type=10)
    elif args.dataset =='logic':
        dataset = ds.BasicLogicDataset(n_per_type=100)
    elif args.dataset == 'brackets':
        dataset = ds.BracketsDataset(n=100)
    elif args.dataset == 'counting':
        dataset = ds.NaturalCountingDataset(n_per_type=20)
    elif args.dataset == 'coding_basics':
        dataset = ds.ProgrammingBasicsDataset(n_per_type=10)
    elif args.dataset == 'string_ops':
        dataset = ds.CoreCapabilitiesDataset(qtypes=["concat_or_slice", "reverse"], n_per_type=30)
    elif args.dataset == 'string_concat_slice':
        dataset = ds.CoreCapabilitiesDataset(qtypes="concat_or_slice", n_per_type=100)
    elif args.dataset == 'string_reverse':
        dataset = ds.CoreCapabilitiesDataset(qtypes="reverse", n_per_type=100)
    elif args.dataset == 'loop_unroll':
        dataset = ds.CoreCapabilitiesDataset(qtypes='loop_unroll', n_per_type=100)
    elif args.dataset == 'indexing':
        dataset = ds.CoreCapabilitiesDataset(qtypes='indexing', n_per_type=100)
    else:
        dataset = ds.ProgrammingSkillsDataset(qtype='all',n_per_type=10)
    if args.num_samples == -1:
        args.num_samples = len(dataset)
    sampled_indices = random.sample(range(len(dataset)), k=args.num_samples)
    message_list = []
    input_strs = []
    label_strs = []
    texts = []
    for i in sampled_indices:
        input_str = dataset[i]["input"]
        label = dataset[i]["target"]
        if args.dataset == 'twodig':
            message_list.append([{"role": "user", "content": "Answer the question directly:\n"+input_str}])
        else:
            message_list.append([{"role": "user", "content": input_str}])
        input_strs.append(input_str)
        label_strs.append(label)
    texts = tokenizer.apply_chat_template(message_list, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(texts, padding="longest", padding_side="left", return_tensors="pt")
    inputs = {key: val.to(model.device) for key, val in inputs.items()}
    input_ids = inputs["input_ids"]
    del texts

    tokens = [clean_print(tokenizer.convert_ids_to_tokens(input_ids[i])) for i in range(len(input_ids))]
    num_new_tokens = 50
    with torch.no_grad():
        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=num_new_tokens, temperature=1.0, top_p=1.0)
    decoded_strs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # Parse the generated answer
    if 'llama' in args.model:
        if args.force_single_digit or 'llama3.2' in args.model:
            decoded_strs = [s.split("assistant\n\n")[-1].split("=")[-1].strip().replace(",", "").strip(".") for s in decoded_strs] 
        else:
            if args.dataset == 'twodig':
                decoded_strs = [decoded_str.split("assistant\n\n")[1] for decoded_str in decoded_strs]
                decoded_strs = [decoded_str.strip(".").replace(",","") for decoded_str in decoded_strs]
            else:
                decoded_strs = [s.partition("assistant\n\n")[2].strip() for s in decoded_strs]
            if args.dataset == 'rhyme':
                decoded_strs = [s.strip(".") for s in decoded_strs]

    elif 'qwen' in args.model:
        decoded_strs = [s.split("assistant\n")[-1].split("=")[-1].strip().replace(",", "").replace("<|endoftext|>", "").strip(".") for s in decoded_strs]
    cor = 0 
    for input_str, decoded_str, label_str in zip(input_strs, decoded_strs, label_strs):
        cor += (decoded_str == label_str)

    acc = cor / len(sampled_indices)
    print(f"acc is {acc}")

    del model, tokenizer, inputs, outputs
    del layers
    del input_strs, decoded_strs, label_strs
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    import gc; gc.collect()

    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset', type=str, default='twodig')
    parser.add_argument("--arithmetic_type", type=str, default="all")
    parser.add_argument('--layer_id', type=int)
    parser.add_argument('--layer_id_list', type=int, default=-1, nargs='+', help='List of integers separated by space')
    parser.add_argument('--attn_head_list', type=int, default=-1, nargs='+', help='List of integers separated by space')
    parser.add_argument('--single_head', action="store_true")

    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--access_token", type=str) 
    
    parser.add_argument("--savedir", type=str, default=".")
    parser.add_argument("--single_run", action="store_true")
    parser.add_argument("--complement", action="store_true")
    parser.add_argument("--force_single_digit", action="store_true")
    parser.add_argument("--mean_ablation", action="store_true")
    parser.add_argument("--patching", action="store_true")
    args = parser.parse_args()
    print(args)

    if args.single_run: # Do one normal run
        main(args)

    else: # Collect accs for ablating each head in turn throughout the network
        model, tokenizer = load_llm_hf(args)
        
        layers = model.config.num_hidden_layers
        heads = model.config.num_attention_heads
        if args.mean_ablation:
            collect_mean_ablation(args, model, heads)
            print('done with mean ablations')
        del model, tokenizer
        all_accs = []
        for layer_id in range(layers):
            print("Layer ", layer_id)
            acc_lst = []
            for head_id in range(heads):
                args.layer_id = layer_id
                args.attn_head_list = [head_id]
                acc = main(args)
                acc_lst.append(acc)
            all_accs.append(acc_lst)
        all_accs = np.array(all_accs)
        modelname = get_name(args)
        if args.complement:
            comp_str = 'complement_'
        else:
            comp_str = ''
        if args.dataset == 'twodig':
            ds_name = f'twodig_{args.arithmetic_type}'
        else:
            ds_name = args.dataset
        np.save(f'{modelname}_{ds_name}_{comp_str}{args.seed}.npy', all_accs)
