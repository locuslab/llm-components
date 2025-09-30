import torch 
import argparse
import os 
import pickle
import random

import numpy as np 
import matplotlib 
import monkey_patch as mp
from lib.model_dict import MODEL_DICT_LLMs
import lib 
import torch.nn.functional as F
import numpy as np
import re 
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

from pdb import set_trace as st

from openai import OpenAI

from input_texts import *
from dataset import TwoDigitArithmeticDataset

import logging as pylogging
pylogging.getLogger("transformers").setLevel(pylogging.ERROR)



def extract_head_info(head_info):
    match = re.match(r'L(\d{1,2})H(\d{1,2})', head_info)
    if match:
        numbers = [int(match.group(1)), int(match.group(2))]
        return numbers
    else:
        raise ValueError("Invalid head_info")

def get_name(args):
    shorthands = ['llama', 'qwen', 'pythia', 'gptj', 'gemma']
    for name in shorthands:
        if name in args.model:
            return name
        
def get_dataset(args):
    # ds_dir = "/data/user_data/abair/huggingface"
    ds_dir = "/data/locus/project_data/project_data3/abair/huggingface"
    if args.dataset == 'lmsys':
        ds = load_dataset("lmsys/lmsys-chat-1m", cache_dir=ds_dir)
    elif args.dataset == 'gsm8k':
        ds = load_dataset("openai/gsm8k", "main", cache_dir=ds_dir)
    elif args.dataset == 'math500':
        ds = load_dataset("HuggingFaceH4/MATH-500", cache_dir=ds_dir)
    return ds


def clean_print(token_list):
    new_list = []
    for word in token_list:
        if word[0] == 'Ġ':
            new_list.append(word[1:])
        else:
            new_list.append(word)
    return new_list


def create_dataset_plots(args):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    ds = get_dataset(args)
    model, tokenizer = lib.load_llm_hf(args)

    math500_indices_with_multiplication = [9, 10, 19, 20, 25, 30, 42, 66] # 30
    gsm_indices_with_multiplication = [4, 6, 15, 18, 21, 23, 25, 30, 31, 34, 37, 39]

    if args.dataset == 'gsm8k':
        indices_with_multiplication = gsm_indices_with_multiplication
    elif args.dataset == 'math500':
        indices_with_multiplication = math500_indices_with_multiplication

    for i in indices_with_multiplication:
        input_text = ds['test'][i]['problem']
        #system_prompt = "You are a skilled mathematician and a logical thinker. Solve the following math problem step by step. If you get stuck, just provide your best guess. Provide the correct answer at the end of your response in the format: Final answer: <solution> " # 81
        # input_text = "What is the result of 9 x 2? 18."
        # input_text = "What is 6 x 6? Answer: 36."
        # input_text = "What is five times three? Fifteen."
        # input_text = "8x3=24 \n5x5=25 \n7x6=42"
        # input_text = ""
        input_ids = tokenizer(input_text, return_tensors='pt').to(device)
        # input_ids = tokenizer.apply_chat_template([{"role": "user", "content": system_prompt + prompt}], tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**input_ids, output_attentions=True)

        tokens = clean_print(tokenizer.convert_ids_to_tokens(input_ids['input_ids'][0]))[1:]

        # for layer in range(19, 32):
        #     print(layer)
        #     for head in range(32):
        layer = 19
        head = 17
        plt.clf()
        attn_outputs = outputs.attentions[layer][:, head, :, :]

        if "pythia" or "qwen" in args.model:
            attn_map = attn_outputs[0].detach().cpu().numpy()
        else:
            attn_map = attn_outputs[0].detach().cpu().numpy()[1:, 1:]

        plots_dir = 'attn_maps_math500'
        os.makedirs(plots_dir, exist_ok=True)

        plt.imshow(attn_map, cmap='viridis')
        plt.colorbar()
        plt.title(f'L{layer}H{head}')
        plt.xticks(range(len(tokens)), tokens, rotation=45)
        plt.yticks(range(len(tokens)), tokens)
        plt.savefig(f'{plots_dir}/{i}_L{layer}H{head}.pdf')


def create_single_plot(args):
    model, tokenizer = lib.load_llm_hf(args)
    
    input_texts, _ = _get_input_texts(args)
    layer, head = extract_head_info(args.head_str)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    for idx, input_text in enumerate(input_texts):
        # import pdb; pdb.set_trace()
        if "pythia" in model.config._name_or_path or "Qwen" in model.config._name_or_path:
            input_text = "<|endoftext|>" + input_text
        input_ids = tokenizer(input_text, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**input_ids, output_attentions=True)
        tokens = clean_print(tokenizer.convert_ids_to_tokens(input_ids['input_ids'][0]))[1:]
        tick_tokens = [label.replace('$', r'\$') for label in tokens]

        attn_outputs = outputs.attentions[layer][:, head, :, :]
        attn_map = attn_outputs[0].detach().to(dtype=torch.float32).cpu().numpy()[1:, 1:]

        os.makedirs(args.savedir, exist_ok=True)
        plt.clf()
        fig, ax = plt.subplots(figsize=(6, 6))
        plot_map = np.where(np.tril(np.ones_like(attn_map)), attn_map, np.nan)
        vmin = -10
        vmax = 256
        original_cmap = plt.cm.Blues
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            'truncated_viridis',
            original_cmap(norm(range(original_cmap.N))),  # Apply normalization
            original_cmap.N
        )
        cmap.set_bad(color='white')

        im = ax.imshow(plot_map, cmap=cmap)
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=16)

        ax.set_title(f'Layer {layer}, Head {head}', fontsize=18, pad=12)
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tick_tokens, fontsize=16)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tick_tokens, fontsize=16)
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.tight_layout()
        image_path = f'{args.savedir}/L{layer}H{head}_{idx}.pdf'
        plt.savefig(image_path, bbox_inches='tight')
        plt.close()


def heatmap_plot(args):
    modelname = 'qwen'
    # import pdb; pdb.set_trace()
    heatmap = np.load(f'{modelname}_{args.list}_{args.tok1}_{args.tok2}_single_digit.npy')
    print(np.max(heatmap))
    import pdb; pdb.set_trace()
    # heatmap1 = np.load(f'{modelname}_addition_4_3_double_digit.npy')
    # heatmap2 = np.load(f'{modelname}_addition_5_4_double_digit.npy')
    # heatmap = heatmap1 + heatmap2
    fig, ax = plt.subplots(figsize=(6, 6))
    # plot_map = np.where(np.tril(np.ones_like(attn_map)), attn_map, np.nan)
    vmin = -10
    vmax = 256
    original_cmap = plt.cm.Blues
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'truncated_viridis',
        original_cmap(norm(range(original_cmap.N))),  # Apply normalization
        original_cmap.N
    )
    cmap.set_bad(color='white')

    im = ax.imshow(heatmap, cmap=cmap)
    # cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    # cbar.ax.tick_params(labelsize=16)
    ax.set_xlabel('Heads')
    ax.set_ylabel('Layers')
    ax.set_title(f'{modelname} {args.list} {args.tok1}/{args.tok2}')
    # ax.set_title(f'{modelname} {args.list} 4/3 + 5/4 double digit')


    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    image_path = f'{args.savedir}/{modelname}_{args.list}_{args.tok1}_{args.tok2}_double_digit.pdf'
    # image_path = f'{args.savedir}/{modelname}_{args.list}_43_54_double_digit.pdf'
    plt.savefig(image_path, bbox_inches='tight')
    plt.close()


def get_loc_tokens(model, tokenizer, input_text, layer, head, idx1, idx2):
    """
    a o b = c
    0 1 2 3 4

    b/o 2/1
    """
    input_text = input_text.replace(' ', '')
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if "pythia" in model.config._name_or_path or "Qwen" in model.config._name_or_path:
        input_text = "<|endoftext|>" + input_text
    elif "gpt-j" in model.config._name_or_path:
        input_text = "<|endoftext|>" + input_text
    input_ids = tokenizer(input_text, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**input_ids, output_attentions=True)
    # tokens = clean_print(tokenizer.convert_ids_to_tokens(input_ids['input_ids'][0]))[1:]
    attn_outputs = outputs.attentions[layer][:, head, :, :]
    attn_map = attn_outputs[0].detach().to(dtype=torch.float32).cpu().numpy()[1:, 1:]
    attn_val = attn_map[idx1, idx2] #- attn_map[0, 0] #################################### subtracted off mean here
    return attn_val


def _get_input_texts(args):
    if args.list == 'addition':
        texts_list = addition_input_texts
        title = 'Addition'
    elif args.list == 'subtraction':
        texts_list = subtraction_input_texts
        title = 'Subtraction'
    elif args.list == 'multiplication':
        texts_list = multiplication_input_texts
        title = 'Multiplication'
    elif args.list == 'division':
        texts_list = division_input_texts
        title = 'Division'
    elif args.list == 'exponentiation':
        texts_list = exponentiation_input_texts
        title = 'Exponentiation'
    return texts_list, title



def _create_heatmap(model, tokenizer, input_texts, tok1, tok2):
    layers = model.config.num_hidden_layers
    heads = model.config.num_attention_heads
    heatmap = []
    for layerid in tqdm(range(layers)):
        headvals = []
        for headid in range(heads):
            mean_attn = 0
            for input_text in input_texts:
                attn_val = get_loc_tokens(model, tokenizer, input_text, layerid, headid, tok1, tok2)
                mean_attn += attn_val
            mean_attn /= len(input_texts)
            headvals.append(mean_attn)
        heatmap.append(headvals)
    heatmap = np.array(heatmap)
    return heatmap


def _create_heatmap_diff_dataset(model, tokenizer, args):
    dataset = TwoDigitArithmeticDataset(arithmetic_type=args.list, modeltype=args.model)
    sampled_indices = random.sample(range(len(dataset)), k=args.num_samples)

    layers = model.config.num_hidden_layers
    heads = model.config.num_attention_heads
    heatmap = []
    for layerid in tqdm(range(layers)):
        headvals = []
        for headid in range(heads):
            mean_attn = 0
            for i in sampled_indices:
                input_text = dataset[i]["input"]
                attn_val = get_loc_tokens(model, tokenizer, input_text, layerid, headid, args.tok1, args.tok2)
                mean_attn += attn_val
            mean_attn /= len(sampled_indices)
            headvals.append(mean_attn)
        heatmap.append(headvals)
    heatmap = np.array(heatmap)
    return heatmap


def _heatmap_processing(heatmap):
    """
    Returns top 10 indexes globally across the heatmap
    """
    flat_indices = np.argsort(heatmap, axis=None)[-10:][::-1]
    row_indices, col_indices = np.unravel_index(flat_indices, heatmap.shape)
    top_coords = list(zip(row_indices, col_indices))
    return top_coords



def single_run(args):
    model, tokenizer = lib.load_llm_hf(args)
    # texts_list, title = _get_input_texts(args)
    #heatmap = _create_heatmap(model, tokenizer, texts_list, args.tok1, args.tok2)
    heatmap = _create_heatmap_diff_dataset(model, tokenizer, args)
    modelname = get_name(args)
    np.save(f'{modelname}_twodig_{args.list}_{args.tok1}_{args.tok2}.npy', heatmap)

    global_sort=True
    if global_sort:
        top_coords = _heatmap_processing(heatmap)
        for c in top_coords:
            print(c, " : ", round(heatmap[c], 4))
    else:
        k=5
        layer_avg = np.mean(heatmap, axis=1)
        max_layer = np.argmax(layer_avg)
        print('Max Layer:', max_layer, round(layer_avg[max_layer], 4))
        idxs = np.argpartition(heatmap[max_layer], -k)[-k:]
        idxs = idxs[np.argsort(heatmap[max_layer][idxs])[::-1]]
        for idx in idxs:
            print(idx, ':', round(heatmap[max_layer][idx], 4))



def get_baselines(args):
    do_avg = True
    model, tokenizer = lib.load_llm_hf(args)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    layers = model.config.num_hidden_layers
    heads = model.config.num_attention_heads
    dataset = TwoDigitArithmeticDataset(arithmetic_type=args.list, modeltype=args.model)
    sampled_indices = random.sample(range(len(dataset)), k=100)
    # input_texts = addition_input_texts + subtraction_input_texts + multiplication_input_texts + division_input_texts
    heatmap = []
    for layer in tqdm(range(layers)):
        headvals = []
        for head in range(heads):
            mean_attn = 0
            # for input_text in input_texts:
            for i in range(len(sampled_indices)):
                input_text = dataset[i]["input"]
                if "pythia" in model.config._name_or_path or "Qwen" in model.config._name_or_path:
                    input_text = "<|endoftext|>" + input_text
                elif args.model == "gptj":
                    input_text = "<|endoftext|>" + input_text
                input_ids = tokenizer(input_text, return_tensors='pt').to(device)
                with torch.no_grad():
                    outputs = model(**input_ids, output_attentions=True)
                # tokens = clean_print(tokenizer.convert_ids_to_tokens(input_ids['input_ids'][0]))[1:]
                attn_outputs = outputs.attentions[layer][:, head, :, :]
                attn_map = attn_outputs[0].detach().to(dtype=torch.float32).cpu().numpy()[1:, 1:]
                if do_avg:
                    num_nonzero_elts = (attn_map.shape[0] * (attn_map.shape[0] + 1)) / 2
                    val = attn_map.sum() / num_nonzero_elts
                else:
                    val = attn_map[args.tok1, args.tok2]
                mean_attn += val 
            mean_attn /= len(sampled_indices)
            headvals.append(mean_attn)
        heatmap.append(headvals)
    heatmap = np.array(heatmap)
    modelname = get_name(args)
    np.save(f'{modelname}_twodig_avg_{args.list}.npy', heatmap)
    
    

def run(args):
    """
    ** How to specify args.tok1 and args.tok2 **

    All input texts are in the format below, with corresponding token indices to be used for arguments tok1 and tok2.
    a o b = c
    0 1 2 3 4

    For example, if you have
    5 + 3 = 8

    and if you want to compute the attention from = to 3, you want tok1 = 3 and tok2 = 2.
    """
    # get_baselines(args)
    if args.run == "plot":
        ### Create heatmaps for a single head's activation on a set of texts
        ### Specify: head_str, list, tok1, tok2, savedir
        create_single_plot(args) 
    elif args.run == "single_run":
        ### Get top activating heads either globally or layerwise for a pair of tokens
        ### Specify: list, tok1, tok2
        single_run(args)
    elif args.run == "baselines":
        ### Get averages across all tokens, all lists, for a given model
        ### Specify: model, (list, token)
        get_baselines(args)
    elif args.run == 'heatmap':
        heatmap_plot(args)
    else:
        print("Specify one of 'plot', 'single_run', 'average_run', or 'delta_run'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model', default='llama3.1-8b-it') # gemma-2-9b-it, qwen2.5-7b-it, pythia-6.9b, gpt2_xl, llama3.1-8b-it
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--access_token", type=str, default="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")
    parser.add_argument("--dataset", type=str, default="math500")

    parser.add_argument("--head_str", type=str, default="L0H0")
    parser.add_argument("--savedir", type=str, default=".")
    parser.add_argument("--list", type=str)
    parser.add_argument("--tok1", type=int)
    parser.add_argument("--tok2", type=int)
    parser.add_argument("--run", type=str, default="single_run")
    parser.add_argument("--delta", action='store_true')
    args = parser.parse_args()
    print(args)

    run(args)
