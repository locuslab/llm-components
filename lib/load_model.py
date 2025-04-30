import os
from vllm import LLM, SamplingParams
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, GPTNeoXForCausalLM, GPT2Tokenizer, GPT2Model

from .model_dict import MODEL_DICT_LLMs 
import logging 

from pdb import set_trace as st

def load_vllm(args):
    print(f"loading model {args.model}")
    model_name, cache_dir = MODEL_DICT_LLMs[args.model]["model_id"], MODEL_DICT_LLMs[args.model]["cache_dir"]
    logging.info(f"loading model {model_name} cache_dir {cache_dir}")

    if args.model == "llama3.1-70b-it":
        print("vllm loading llama3.1-70b-it")
        print("cuda available ", torch.cuda.is_available())
        print("device_count ", torch.cuda.device_count())
        print("cuda_visible_devices ", os.environ["CUDA_VISIBLE_DEVICES"])
        num_gpus = torch.cuda.device_count()
        model = LLM(model_name, download_dir=cache_dir, max_model_len=2048, tensor_parallel_size=num_gpus, dtype="bfloat16")
        # model = LLM(model_name, download_dir=cache_dir, max_model_len=1024, tensor_parallel_size=4, dtype="auto")
    # elif "gemma" in model_name:
    #     model = LLM(model_name, download_dir=cache_dir)
    else:
        model = LLM(model_name, download_dir=cache_dir, dtype="bfloat16")
    tokenizer = model.get_tokenizer()

    return model, tokenizer 

def load_llm_hf(args):
    print(f"loading model {args.model}")
    model_name, cache_dir = MODEL_DICT_LLMs[args.model]["model_id"], MODEL_DICT_LLMs[args.model]["cache_dir"]

    if "llama3-" in args.model:
        config = AutoConfig.from_pretrained(cache_dir)
        kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
        model = AutoModelForCausalLM.from_pretrained(cache_dir, config=config, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, token=args.access_token, trust_remote_code=True)
    elif "llama3.1-" in args.model:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, cache_dir=cache_dir, low_cpu_mem_usage=True, device_map="auto", token=args.access_token)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, token=args.access_token, trust_remote_code=True)
    elif "gemma" in args.model:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, cache_dir=cache_dir, low_cpu_mem_usage=True, device_map="auto", token=args.access_token)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, token=args.access_token, cache_dir=cache_dir, trust_remote_code=True)
    elif "pythia" in args.model:
        model = GPTNeoXForCausalLM.from_pretrained(model_name, revision="step3000", cache_dir=cache_dir, device_map="auto").to('cuda')
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision="step3000", cache_dir=cache_dir)
    elif "qwen" in args.model:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    elif "gpt2" in args.model:
        model = GPT2Model.from_pretrained('gpt2-xl', torch_dtype="auto", device_map="auto", cache_dir=cache_dir)
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl', cache_dir=cache_dir, add_bos_token=True)


    return model, tokenizer 

def step2revision(step):
    from huggingface_hub import HfApi
    api = HfApi()
    refs=api.list_repo_refs("allenai/OLMo-7B")
    total_list = []
    for branch in refs.branches:
        name = branch.name
        total_list.append(name)

    ##############################################
    mapping = {}
    # Iterate over each string in the list
    for s in total_list:
        # Extract the number (assumes the format 'step<number>-tokens<something>')
        number = s.split('-')[0].replace('step', '')
        # Add to the dictionary with the number as key and the string as value
        mapping[number] = s

    return mapping[step]
