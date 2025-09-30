### May need to be moved to lm-evaluation-harness dir depending on setup
import argparse
from pathlib import Path
from collections import defaultdict
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)) 
from modify_model import zero_ablate_attn_head, amplify_attn_head

import lm_eval
from lm_eval.tasks import TaskManager
from lm_eval.models.huggingface import HFLM

import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

TOKENIZER_DIR = None 
DEVICE = "cuda:0"
DTYPE = "bfloat16"


def get_model_dir(args):
    if args.model == 'llama3.1-8b-it':
        model_dir = "/data/locus/project_data/project_data3/abair/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659" 
    elif args.model == 'qwen2.5-3b-it':
        model_dir = "/data/locus/project_data/project_data3/abair/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
    elif args.model == 'qwen2.5-7b-it':
        model_dir = "/data/locus/project_data/project_data3/abair/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    elif args.model == 'qwen2.5-14b-it':
        model_dir = "/data/locus/project_data/project_data3/abair/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
    elif args.model == 'llama3.2-3b-it':
        model_dir = "/data/locus/project_data/project_data3/abair/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95"
    elif args.model == 'llama3.2-1b-it':
        model_dir = "/data/locus/project_data/project_data3/abair/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6"
    else:
        print("Misspecified model")
        return
    return model_dir


def run_lm_eval(args):
    model_dir = get_model_dir(args)
    lm = HFLM(
        pretrained=model_dir,
        tokenizer=TOKENIZER_DIR, 
        device=DEVICE,
        dtype=DTYPE,
        trust_remote_code=True, 
        parallelize=False, 
        batch_size="auto",
    )
    if args.zero_ablate:
        layers = lm.model.model.layers
        if args.layer_id_list != -1:
            groups = defaultdict(list)
            for l, h in zip(args.layer_id_list, args.attn_head_list):
                groups[l].append(h)
            pairs =  [(l, hs) for l, hs in groups.items()]
            for layer_idx, head_lst in pairs:
                zero_ablate_attn_head(layers[layer_idx], head_lst)

        else:
            zero_ablate_attn_head(layers[args.layerid], args.attn_head_list)

    elif args.amplify:
        layers = lm.model.model.layers
        if args.layer_id_list != -1:
            groups = defaultdict(list)
            for l, h in zip(args.layer_id_list, args.attn_head_list):
                groups[l].append(h)
            pairs =  [(l, hs) for l, hs in groups.items()]
            for layer_idx, head_lst in pairs:
                amplify_attn_head(layers[layer_idx], head_lst)

        else:
            amplify_attn_head(layers[args.layerid], args.attn_head_list)


    tm = TaskManager() 
    if args.task == 'asdiv_cot_llama':
        shot = 0
    else:
        shot = None
    res = lm_eval.simple_evaluate(
        model=lm,
        tasks=[args.task], 
        num_fewshot=shot,
        batch_size="auto",
        task_manager=tm,
        confirm_run_unsafe_code=True,
    )

    for task, metrics in res["results"].items():
        print(task, {k:v for k,v in metrics.items()})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--zero_ablate", action="store_true")
    parser.add_argument("--amplify", action="store_true")
    parser.add_argument("--layerid", type=int)
    parser.add_argument('--attn_head_list', type=int, default=-1, nargs='+', help='List of integers separated by space')
    parser.add_argument('--layer_id_list', type=int, default=-1, nargs='+', help='List of integers separated by space')
    parser.add_argument('--shot', type=int, default=0)
    parser.add_argument('--model', type=str, default='llama3.1-8b-it')
    args = parser.parse_args()
    print(args)
    run_lm_eval(args)

if __name__ == "__main__":
    main()
