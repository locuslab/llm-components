import torch 
import argparse
import os 

import numpy as np 
import monkey_patch as mp
from lib.model_dict import MODEL_DICT_LLMs
import lib 
import torch.nn.functional as F
import numpy as np
import re 
from datasets import load_dataset
from tqdm import tqdm

from openai import OpenAI

from pdb import set_trace as st 

import logging as pylogging
pylogging.getLogger("transformers").setLevel(pylogging.ERROR)


def get_dataset(args):
    ds_dir = "/data/user_data/abair/huggingface"
    # ds_dir = "/data/locus/project_data/project_data3/abair/huggingface"
    if args.dataset == 'lmsys':
        ds = load_dataset("lmsys/lmsys-chat-1m", cache_dir=ds_dir)
    elif args.dataset == 'gsm8k':
        ds = load_dataset("openai/gsm8k", "main", cache_dir=ds_dir)
    elif args.dataset == 'math500':
        ds = load_dataset("HuggingFaceH4/MATH-500", cache_dir=ds_dir)
    return ds


def extract_head_info(head_info):
    match = re.match(r'L(\d{1,2})H(\d{1,2})', head_info)
    if match:
        numbers = [int(match.group(1)), int(match.group(2))]
        return numbers
    else:
        raise ValueError("Invalid head_info")


def get_loss(input_ids, model, tokenizer, prompt, device = torch.device("cuda:0")):
    with torch.no_grad():
        outputs = model(input_ids)

    logits = outputs.logits 
    logits = logits.view(-1, logits.shape[-1])[:-1,:]
    labels = input_ids.view(-1)[1:]
    loss = F.cross_entropy(logits, labels, reduction="none")
    return loss, outputs


def clean_print(token_list):
    new_list = []
    for word in token_list:
        if word[0] == 'Ġ':
            new_list.append(word[1:])
        else:
            new_list.append(word)
    print(new_list)


def get_non_ablated_losses(args, ds, model, tokenizer, device=torch.device("cuda:0")):
    if not os.path.exists(os.path.join(args.savedir, f"non_ablated_{args.dataset}.pt")):
        obj = acc_and_token_loss(args, model, tokenizer, ds, device)
        torch.save(obj, os.path.join(args.savedir, f"non_ablated_{args.dataset}.pt"))


def collect_mean_ablation(args, model):
    """
    Collect mean head statistics across the whole model
    Saves mean values in mean_outputs_{num_samples}.pt
    """
    layers = model.model.layers
    for layer_id in range(len(layers)):
        mp.mean_ablate_llama3_attn_head(layers[layer_id], 0, collect_stats=True)
    outputs = []
    for layer_id in range(len(layers)):
        mean_attn_outputs = layers[layer_id].self_attn.mean_head_stats
        outputs.append(mean_attn_outputs)
    torch.save(outputs, os.path.join(args.savedir, f"mean_outputs_{args.num_samples}.pt"))


def extract_final_answer(response, dataset):
    if dataset == 'gsm8k':
        # Example: extract last number (modify as needed)
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", response)
        return nums[-1] if nums else None
    elif dataset == 'math500':
        match = re.findall(r"Final answer:\s*(.*)", response)
        return match[-1] if match else None

def chatgpt_correct(groundtruth_answer, response, dataset):
    client = OpenAI(
        api_key='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',  # This is the default and can be omitted
    )
    prompt = f"You are a highly skilled mathematician. You must compare pairs of expressions that may be formatted differently and determine if the expressions represent the same mathematical quantity. Are the following two expressions mathematically equivalent? Respond with a single word answer, yes or no. Expression 1: {groundtruth_answer}, Expression 2: {extract_final_answer(response, dataset)}"
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4o-mini",
    )
    return chat_completion.choices[0].message.content.lower() == 'yes'

def acc_and_token_loss(args, model, tokenizer, ds, get_acc=False, device=torch.device("cuda:0")):
    obj = []
    for i in tqdm(range(args.num_samples)):
        if args.dataset == 'lmsys':
            prompt = ds["train"][i]["conversation"][0]["content"]
        elif args.dataset == 'gsm8k':
            prompt = ds['train'][i]['question']
            system_prompt = "You are a helpful assistant. At the end of your answer, provide the numerical answer without unnecessary symbols or unused decimal places in the format: Final answer: <number> "
        elif args.dataset == 'math500':
            prompt = ds['test'][i]['problem']
            system_prompt = "You are a skilled mathematician and a logical thinker. Solve the following math problem step by step. If you get stuck, just provide your best guess. Provide the correct answer at the end of your response in the format: Final answer: <solution> " # 81
            # system_prompt = "You are a math solver. Provide a concise solution within 1000 tokens. Summarize any extra detail briefly. Provide your final answer at the end of your response in the format: Final answer: <answer> "
        input_ids = tokenizer.apply_chat_template([{"role": "user", "content": system_prompt + prompt}], tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
        loss, outputs = get_loss(input_ids, model, tokenizer, prompt)
        if get_acc:
            # output_ids is from generated tokens
            output_ids = model.generate(input_ids, max_new_tokens=2048)
            response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            if args.dataset == 'gsm8k':
                groundtruth_answer = ds['train'][i]['answer'].split('####')[-1].replace(" ", "")
                correct = groundtruth_answer == extract_final_answer(response, args.dataset)
            elif args.dataset == 'math500':
                groundtruth_answer = ds['test'][i]['answer']
                correct = chatgpt_correct(groundtruth_answer, response, args.dataset) 
        else:
            correct = None
        obj.append({"prompt": prompt, "loss": loss.cpu(), "correct": correct, 'input_ids': input_ids})
    return obj


def mean_ablation(args, model, tokenizer, ds, device = torch.device("cuda:0")):
    """
    Performs the mean ablation for a given attention head. 
    Finds the per-token loss and the accuracy of the head-ablated model for each sample
    Saves loss and accuracy per sample in mean_ablate_{head}.pt
    """
    if not os.path.exists(os.path.join(args.savedir, f"mean_ablate_{args.head_str}.pt")):
        layers = model.model.layers
        layer_id, attn_id = extract_head_info(args.head_str)
        obj = torch.load(os.path.join(args.savedir, f"mean_outputs_{args.num_samples}.pt"), weights_only=True)
        layers[layer_id].self_attn.mean_head_stats = obj[layer_id]
        mp.mean_ablate_llama3_attn_head(layers[layer_id], attn_id, collect_stats=False)

        obj = acc_and_token_loss(args, model, tokenizer, ds)
        torch.save(obj, os.path.join(args.savedir, f"mean_ablate_{args.head_str}.pt"))


def analyze(args, tokenizer, delta_selection='top5%', sentences=True, acc=True, device=torch.device("cuda:0")):
    # model, tokenizer = lib.load_llm_hf(args)
    obj_default = torch.load(os.path.join(args.savedir, f"non_ablated_{args.dataset}.pt"), weights_only=True)
    obj_mean = torch.load(os.path.join(args.savedir, f"mean_ablate_{args.head_str}.pt"), weights_only=True)
    head_deltas = []

    if args.dataset == 'gsm8k':
        start_idx = 63
    elif args.dataset == 'math500':
        start_idx = 80

    # collect aggregate deltas
    all_deltas = []
    for i in range(len(obj_default)):
        default_loss = obj_default[i]['loss']
        mean_ablated_loss = obj_mean[i]['loss']
        
        deltas = mean_ablated_loss - default_loss 
        deltas = deltas[start_idx:-5].tolist()
        all_deltas.extend(deltas)

    top_percent = np.quantile(all_deltas, 0.95)

    default_acc = 0
    ablated_acc = 0
    sentences_lst = []
    for i in range(len(obj_default)):
        input_ids = obj_default[i]['input_ids']
        default_loss = obj_default[i]['loss']
        mean_ablated_loss = obj_mean[i]['loss']
        deltas = mean_ablated_loss - default_loss 
        labels = input_ids.view(-1)[1:]
        tokens = tokenizer.convert_ids_to_tokens(labels, skip_special_tokens=False)
        prompt_only_deltas = deltas[start_idx:-5]
        indices = np.where(prompt_only_deltas > top_percent)[0]
        prompt_tokens = tokens[start_idx:-5]
        for idx in indices:
            if prompt_tokens[idx][0] == "Ġ":
                prompt_tokens[idx] = prompt_tokens[idx][0] + "{{" + prompt_tokens[idx][1:] + "}}" 
            else:
                prompt_tokens[idx] = "{{" + prompt_tokens[idx] + "}}"
        sentence = "".join(prompt_tokens).replace("Ġ", " ") 
        # print(sentence)
        sentences_lst.append(sentence)
    #     default_correct = obj_default[i]['correct']
    #     mean_ablated_correct = obj_mean[i]['correct']
    #     default_acc += default_correct
    #     ablated_acc += mean_ablated_correct
    # default_acc /= len(obj_default)
    # ablated_acc /= len(obj_default)
    # print(f" {args.head_str} Default acc: ", default_acc)
    # print(f" {args.head_str} Ablated acc: ", ablated_acc)
    obj = {'sentences': sentences_lst, 'default_acc': default_acc, 'ablated_acc': ablated_acc}
    torch.save(obj, os.path.join(args.savedir, f"sentences_{args.head_str}.pt")) 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument("--head_str", type=str, default="L0H0")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--collect_mean", action="store_true")
    parser.add_argument("--delta_analysis", action="store_true")
    parser.add_argument("--access_token", type=str, default="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")
    parser.add_argument("--savedir", type=str, default=".")
    parser.add_argument("--dataset", type=str, default="gsm8k")
    args = parser.parse_args()

    ds = get_dataset(args)

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    for i in range(32):
        for j in range(32):
            args.head_str = f'L{i}H{j}'

            model, tokenizer = lib.load_llm_hf(args)

            print('Doing head ', args.head_str)
            get_non_ablated_losses(args, ds, model, tokenizer)
            collect_mean_ablation(args, model)
            mean_ablation(args, model, tokenizer, ds)
            analyze(args, tokenizer,)


    # if args.collect_mean:
    #     # Collect mean ablations
    #     collect_mean_ablation(args, model)
    #     return
    # elif args.delta_analysis:
    #     # Do analysis of highest activating tokens
    #     delta_analysis(args, ds)
    # else:
    #     # Perfom mean ablation and analyze results
    #     mean_ablation(args, model, tokenizer, ds)
    #     analyze(args, tokenizer)

if __name__ == '__main__':
    main()