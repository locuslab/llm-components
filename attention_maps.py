import torch 
import argparse
import os 
import pickle

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

from openai import OpenAI

import logging as pylogging
pylogging.getLogger("transformers").setLevel(pylogging.ERROR)


addition_input_texts = [
    "2+3=5",
    "4+5=9",
    "6+7=13",
    "8+9=17",
    "10+4=14",
    "9+5=14",
    "12+3=15",
    "11+5=16",
    "3+7=10",
    "15+6=21"
]
subtraction_input_texts = [
    "7-3=4",
    "9-4=5",
    "10-6=4",
    "15-5=10",
    "12-9=3",
    "6-2=4",
    "18-9=9",
    "20-8=12",
    "25-5=20",
    "14-7=7"
]
multiplication_input_texts = [
    "2x3=6",
    # "2 \\cdot 3 = 6",
    "4*5=20",
    "6x7=42",
    "8*9=72",
    # "8 \\cdot 9 = 72",
    "7x5=35",
    "9*6=54",
    # "9 \\cdot 6 = 54",
    "10x4=40",
    # "10 \\cdot 4 = 40",
    "12*3=36",
    "11x5=55",
    "14*7=98"
] 


multiplication_3_input_texts = [
  "2x3x4=24",
  "5*6*2=60",
  "1x7x8=56",
  "3*3*3=27",
  "4x2x5=40",
  "6*1*9=54",
  "7x2x3=42",
  "8*4*1=32",
  "9x2x2=36",
  "10*1*3=30"
]

# multiplication_input_texts = [
#     "2x3=6",
#     # "2 \\cdot 3 = 6",
#     "4x5=20",
#     "6x7=42",
#     "8x9=72",
#     # "8 \\cdot 9 = 72",
#     "7x5=35",
#     "9x6=54",
#     # "9 \\cdot 6 = 54",
#     "10x4=40",
#     # "10 \\cdot 4 = 40",
#     "12x3=36",
#     "11x5=55",
#     "14x7=98"
# ] 


# division_input_texts = [
#     "6÷3=2",
#     "8/4=2",
#     "12÷3=4",
#     "9/3=3",
#     "16÷4=4",
#     "20/5=4",
#     "21÷7=3",
#     "24/6=4",
#     "30÷5=6",
#     "36/6=6"
# ]


division_input_texts = [
    "6/3=2",
    "8/4=2",
    "12/3=4",
    "9/3=3",
    "16/4=4",
    "20/5=4",
    "21/7=3",
    "24/6=4",
    "30/5=6",
    "36/6=6"
]

exponentiation_input_texts = [
    # "2**3=8",
    "5**2=25",
    # "10**1=10",
    "3**4=81",
    # "6**2=36",
    "4**3=64",
    # "7**2=49",
    "9**2=81",
    # "2**5=32",
    "8**2=64"
    "2^3=8",
    # "5^2=25",
    "10^1=10",
    # "3^4=81",
    "6^2=36",
    # "4^3=64",
    "7^2=49",
    # "9^2=81",
    "2^5=32",
    # "8^2=64"
]
    

def extract_head_info(head_info):
    match = re.match(r'L(\d{1,2})H(\d{1,2})', head_info)
    if match:
        numbers = [int(match.group(1)), int(match.group(2))]
        return numbers
    else:
        raise ValueError("Invalid head_info")


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

        input_ids = tokenizer(input_text, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**input_ids, output_attentions=True)

        tokens = clean_print(tokenizer.convert_ids_to_tokens(input_ids['input_ids'][0]))[1:]
        tick_tokens = [label.replace('$', r'\$') for label in tokens]

        attn_outputs = outputs.attentions[layer][:, head, :, :]
        attn_map = attn_outputs[0].detach().cpu().numpy()[1:, 1:]

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
        ax.set_xticklabels(tick_tokens, rotation=45, ha='right', fontsize=16)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tick_tokens, fontsize=16)
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.tight_layout()
        image_path = f'{args.savedir}/L{layer}H{head}_{idx}.pdf'
        plt.savefig(image_path, bbox_inches='tight')
        plt.close()


def get_max_tokens(model, tokenizer, input_text, layer, head):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    input_ids = tokenizer(input_text, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**input_ids, output_attentions=True)

    attn_outputs = outputs.attentions[layer][:, head, :, :]
    attn_map = attn_outputs[0].detach().cpu().numpy()[1:, 1:]

    return np.max(attn_map)


def get_loc_tokens(model, tokenizer, input_text, layer, head, idx1, idx2):
    """
    a o b = c
    0 1 2 3 4

    b/o 2/1
    """
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    input_ids = tokenizer(input_text, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**input_ids, output_attentions=True)

    # tokens = clean_print(tokenizer.convert_ids_to_tokens(input_ids['input_ids'][0]))[1:]
    # import pdb; pdb.set_trace()

    attn_outputs = outputs.attentions[layer][:, head, :, :]
    attn_map = attn_outputs[0].detach().cpu().numpy()[1:, 1:]
    attn_val = attn_map[idx1, idx2]
    return attn_val

def get_means(model, tokenizer, layer, head):

    addition_max_tokens_list = [get_max_tokens(model, tokenizer, text, layer, head) for text in addition_input_texts]
    addition_mean = np.mean(addition_max_tokens_list)

    subtraction_max_tokens_list = [get_max_tokens(model, tokenizer, text, layer, head) for text in subtraction_input_texts]
    subtraction_mean = np.mean(subtraction_max_tokens_list)
    
    multiplication_max_tokens_list = [get_max_tokens(model, tokenizer, text, layer, head) for text in multiplication_input_texts]
    multiplication_mean = np.mean(multiplication_max_tokens_list)

    division_max_tokens_list = [get_max_tokens(model, tokenizer, text, layer, head) for text in division_input_texts]
    division_mean = np.mean(division_max_tokens_list)
    return addition_mean, subtraction_mean, multiplication_mean, division_mean


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
    heatmap = []
    for layerid in tqdm(range(32)):
        headvals = []
        for headid in range(32):
            mean_attn = 0
            for input_text in input_texts:
                attn_val = get_loc_tokens(model, tokenizer, input_text, layerid, headid, tok1, tok2)
                mean_attn += attn_val
            mean_attn /= len(addition_input_texts)
            headvals.append(mean_attn)
        heatmap.append(headvals)
    heatmap = np.array(heatmap)
    return heatmap


def _heatmap_processing(heatmap):
    flat_indices = np.argsort(heatmap, axis=None)[-10:][::-1]
    row_indices, col_indices = np.unravel_index(flat_indices, heatmap.shape)
    top_coords = list(zip(row_indices, col_indices))
    return top_coords
    

def get_deltas(add_heatmap, sub_heatmap, mul_heatmap, div_heatmap, exp_heatmap):
    non_add_avg = np.mean([sub_heatmap, mul_heatmap, div_heatmap, exp_heatmap], axis=0)
    non_sub_avg = np.mean([add_heatmap, mul_heatmap, div_heatmap, exp_heatmap], axis=0)
    non_mul_avg = np.mean([add_heatmap, sub_heatmap, div_heatmap, exp_heatmap], axis=0)
    non_div_avg = np.mean([add_heatmap, sub_heatmap, mul_heatmap, exp_heatmap], axis=0)
    non_exp_avg = np.mean([add_heatmap, sub_heatmap, mul_heatmap, div_heatmap], axis=0)

    add_delta = add_heatmap - non_add_avg
    sub_delta = sub_heatmap - non_sub_avg
    mul_delta = mul_heatmap - non_mul_avg
    div_delta = div_heatmap - non_div_avg
    exp_delta = exp_heatmap - non_exp_avg

    # Largest delta
    top_coords = _heatmap_processing(add_delta)
    print('add: ', [(c, round(add_delta[c], 4)) for c in top_coords])
    top_coords = _heatmap_processing(sub_delta)
    print('sub: ', [(c, round(sub_delta[c], 4)) for c in top_coords])
    top_coords = _heatmap_processing(mul_delta)
    print('mul: ', [(c, round(mul_delta[c], 4)) for c in top_coords])
    top_coords = _heatmap_processing(div_delta)
    print('div: ', [(c, round(div_delta[c], 4)) for c in top_coords])
    top_coords = _heatmap_processing(exp_delta)
    print('exp: ', [(c, round(exp_delta[c], 4)) for c in top_coords])

    # Largest absolute value delta
    top_coords = _heatmap_processing(np.abs(add_delta))
    print('add abs: ', [(c, round(add_delta[c], 4)) for c in top_coords])
    top_coords = _heatmap_processing(np.abs(sub_delta))
    print('sub abs: ', [(c, round(sub_delta[c], 4)) for c in top_coords])
    top_coords = _heatmap_processing(np.abs(mul_delta))
    print('mul abs: ', [(c, round(mul_delta[c], 4)) for c in top_coords])
    top_coords = _heatmap_processing(np.abs(div_delta))
    print('div abs: ', [(c, round(div_delta[c], 4)) for c in top_coords])
    top_coords = _heatmap_processing(np.abs(exp_delta))
    print('exp abs: ', [(c, round(exp_delta[c], 4)) for c in top_coords])


def single_run(args):

    model, tokenizer = lib.load_llm_hf(args)
    texts_list, title = _get_input_texts(args)
    heatmap = _create_heatmap(model, tokenizer, texts_list, args.tok1, args.tok2)

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

    plot = False
    if plot:
        plt.imshow(heatmap)
        plt.ylabel('Layer')
        plt.xlabel('Head')
        plt.colorbar()
        plt.title(f'{title} (=/o)')
        plt.savefig(f'{args.list}.pdf')


def means_plot():
    # get means across text list and save
    model, tokenizer = lib.load_llm_hf(args)
    all_means = {'addition': {}, 'subtraction': {}, 'multiplication': {}, 'division': {}}
    for layer in range(32):
        print(layer)
        for head in range(32):
            add, sub, mult, div = get_means(model, tokenizer, layer, head)
            all_means['addition'][f'L{layer}H{head}'] = add
            all_means['subtraction'][f'L{layer}H{head}'] = sub
            all_means['multiplication'][f'L{layer}H{head}'] = mult
            all_means['division'][f'L{layer}H{head}'] = div
    with open("means.pkl", "wb") as f:
        pickle.dump(all_means, f)

    # Do the actual plotting of each head throughout the entire network, print out those that exceed some threshold
    with open("means.pkl", "rb") as f:
        means = pickle.load(f)
    threshold = 0.5
    add = []
    sub = []
    mult = []
    div = []
    for head in means['addition']:
        if means['addition'][head] > threshold:
            add.append(head)
            layerid, headid = extract_head_info(head)
            create_single_plot(model, tokenizer, addition_input_texts, layerid, headid, 'addition')
    for head in means['subtraction']:
        if means['subtraction'][head] > threshold:
            sub.append(head)
            layerid, headid = extract_head_info(head)
            create_single_plot(model, tokenizer, subtraction_input_texts, layerid, headid, 'subtraction')
    for head in means['multiplication']:
        if means['multiplication'][head] > threshold:
            mult.append(head)
            layerid, headid = extract_head_info(head)
            create_single_plot(model, tokenizer, multiplication_input_texts, layerid, headid, 'multiplication')
    for head in means['division']:
        if means['division'][head] > threshold:
            div.append(head)
            layerid, headid = extract_head_info(head)
            create_single_plot(model, tokenizer, division_input_texts, layerid, headid, 'division')
    import pdb; pdb.set_trace()



def delta_run(args):
    model, tokenizer = lib.load_llm_hf(args)
    add_heatmap = _create_heatmap(model, tokenizer, addition_input_texts, args.tok1, args.tok2)
    print('finished addition')
    sub_heatmap = _create_heatmap(model, tokenizer, subtraction_input_texts, args.tok1, args.tok2)
    print('finished subtraction')
    mul_heatmap = _create_heatmap(model, tokenizer, multiplication_input_texts, args.tok1, args.tok2)
    print('finished multiplication')
    div_heatmap = _create_heatmap(model, tokenizer, division_input_texts, args.tok1, args.tok2)
    print('finished division')
    exp_heatmap = _create_heatmap(model, tokenizer, exponentiation_input_texts, args.tok1, args.tok2)
    print('finished exponentiation')

    get_deltas(add_heatmap, sub_heatmap, mul_heatmap, div_heatmap, exp_heatmap)
    

def averaged_run(args):
    """
    This is for finding the attentions between =/c, =/b, and =/a for multiplication problems with 3 operands
    """

    model, tokenizer = lib.load_llm_hf(args)

    mult_heatmap_1 = []
    mult_heatmap_2 = []
    mult_heatmap_3 = []
    for layerid in tqdm(range(32)):
        headvals_1 = []
        headvals_2 = []
        headvals_3 = []
        for headid in range(32):
            mean_attn_1 = 0
            mean_attn_2 = 0
            mean_attn_3 = 0
            for input_text in multiplication_3_input_texts:
                attn_val_1 = get_loc_tokens(model, tokenizer, input_text, layerid, headid, 5, 4)
                attn_val_2 = get_loc_tokens(model, tokenizer, input_text, layerid, headid, 5, 2) 
                attn_val_3 = get_loc_tokens(model, tokenizer, input_text, layerid, headid, 5, 0)
                mean_attn_1 += attn_val_1
                mean_attn_2 += attn_val_2
                mean_attn_3 += attn_val_3
            mean_attn_1 /= len(multiplication_3_input_texts)
            mean_attn_2 /= len(multiplication_3_input_texts)
            mean_attn_3 /= len(multiplication_3_input_texts)
            headvals_1.append(mean_attn_1)
            headvals_2.append(mean_attn_2)
            headvals_3.append(mean_attn_3)
        mult_heatmap_1.append(headvals_1)
        mult_heatmap_2.append(headvals_2)
        mult_heatmap_3.append(headvals_3)
    import pdb; pdb.set_trace()


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
    if args.run == "plot":
        ### Create heatmaps for a single head's activation on a set of texts
        ### Specify: head_str, list, tok1, tok2, savedir
        create_single_plot(args) 
    elif args.run == "single_run":
        ### Get top activating heads either globally or layerwise for a pair of tokens
        ### Specify: list, tok1, tok2
        single_run(args)
    elif args.run == "average_run":
        #TODO: remove some of the hardcoding if needed
        ### Average the activations from =/c, =/b, =/a. Currently just obtains the three heatmaps and hits a breakpoint.
        ### Specify: hardcode token ids in the function, hardcoded input texts
        averaged_run(args)
    elif args.run == "delta_run":
        ### Find the differences between activations on a list of texts - mean([other lists of texts])
        ### Specify: tok1, tok2
        delta_run(args)
    else:
        print("Specify one of 'plot', 'single_run', 'average_run', or 'delta_run'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model', default='llama3.1-8b-it')
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--access_token", type=str, default="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")
    parser.add_argument("--dataset", type=str, default="math500")

    parser.add_argument("--head_str", type=str, default="L0H0")
    parser.add_argument("--savedir", type=str, default=".")
    parser.add_argument("--list", type=str)
    parser.add_argument("--tok1", type=int)
    parser.add_argument("--tok2", type=int)
    parser.add_argument("--run", type=str, default="single_run")
    args = parser.parse_args()
    print(args)

    run(args)
