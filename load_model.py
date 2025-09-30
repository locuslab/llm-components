import os
from vllm import LLM, SamplingParams
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, GPTNeoXForCausalLM, GPT2Tokenizer, GPT2Model

import logging 

from pdb import set_trace as st

CACHE_DIR_BASE = "/data/locus/project_data/project_data3/abair"


MODEL_DICT_LLMs = {

    "qwen2.5-7b-it": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "cache_dir": CACHE_DIR_BASE,
    }, 

    "qwen2.5-3b-it": {
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "cache_dir": CACHE_DIR_BASE,
    }, 

    "qwen2.5-14b-it": {
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
        "cache_dir": CACHE_DIR_BASE,
    }, 

    "llama3.2-1b-it": {
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "cache_dir": CACHE_DIR_BASE,
    },
    "llama3.2-3b-it": {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "cache_dir": CACHE_DIR_BASE,
    },

    "llama3.1-8b-it": {
        "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "cache_dir": CACHE_DIR_BASE,
    },
}


def load_llm_hf(args):
    model_name, cache_dir = MODEL_DICT_LLMs[args.model]["model_id"], MODEL_DICT_LLMs[args.model]["cache_dir"]
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, cache_dir=cache_dir, low_cpu_mem_usage=True, device_map="auto", token=args.access_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, cache_dir=cache_dir, token=args.access_token, trust_remote_code=True)
    return model, tokenizer 
