import os
import torch 
import argparse
import time
import logging
import warnings
import numpy as np 
from collections import defaultdict
from typing import Tuple, List

import lm_eval
from lm_eval.models.huggingface import HFLM

import datasets
datasets.config.HF_DATASETS_OFFLINE = True

os.environ["HF_ALLOW_CODE_EVAL"] = "1"


from load_model import load_llm_hf
from modify_model import zero_ablate_attn_head
from evals import swearing_eval, rhyming_eval, lmeval_evaluate, get_samples

warnings.filterwarnings("ignore")
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


NON_LMEVAL_TASKS = ['swearing', 'rhyming', 'counting']


def get_model_dir(modelname):
    if modelname == 'llama3.1-8b-it':
        model_dir = "/data/locus/project_data/project_data3/abair/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659" 
    elif modelname== 'qwen2.5-3b-it':
        model_dir = "/data/locus/project_data/project_data3/abair/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
    elif modelname == 'qwen2.5-7b-it':
        model_dir = "/data/locus/project_data/project_data3/abair/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    elif modelname == 'qwen2.5-14b-it':
        model_dir = "/data/locus/project_data/project_data3/abair/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
    elif modelname == 'llama3.2-3b-it':
        model_dir = "/data/locus/project_data/project_data3/abair/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95"
    elif modelname == 'llama3.2-1b-it':
        model_dir = "/data/locus/project_data/project_data3/abair/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6"
    else:
        print("Misspecified model")
        return
    return model_dir


def do_zero_ablation(layers, layerid, headid):
    if layerid is not None:
        if isinstance(layerid, list):
            groups = defaultdict(list)
            for l, h in zip(layerid, headid):
                groups[l].append(h)
            pairs = [(l, hs) for l, hs in groups.items()]
            for layer_idx, head_lst in pairs:
                print(f"Ablating Layer {layer_idx}, Head {head_lst}")
                zero_ablate_attn_head(layers[layer_idx], head_lst)
        else:
            print(f"Ablating Layer {layerid} Head {headid}")
            zero_ablate_attn_head(layers[layerid], headid)


def setup_and_ablate(args, layerid=None, headid=None):
    if args.lmeval:
        model_dir = get_model_dir(args.model)
        model = lm_eval.models.huggingface.HFLM(
            pretrained=model_dir,
            tokenizer=None, 
            device='cuda:0',
            dtype='bfloat16',
            trust_remote_code=True, 
            parallelize=False, 
            batch_size=1,
        )
        tokenizer = None
        layers = model.model.model.layers
    else:
        model, tokenizer = load_llm_hf(args)
        model.config.use_cache = False
        device = torch.device("cuda:0")
        model.to(device)
        tokenizer.pad_token_id = tokenizer.eos_token_id 
        layers = model.model.layers
    if args.extra_layers is not None:
        ablate_layers = args.extra_layers.copy()
        ablate_layers.append(layerid)
        ablate_heads = args.extra_heads.copy()
        ablate_heads.extend(headid)
        do_zero_ablation(layers, ablate_layers, ablate_heads)
    else:
        do_zero_ablation(layers, layerid, headid)
    return model, tokenizer


def task_eval(model, tokenizer, task, sampled_examples, num_samples=None):
    if task == 'swearing':
        return swearing_eval(model, tokenizer, num_samples)
    elif task == 'rhyming':
        return rhyming_eval(model, tokenizer, num_samples)
    else:
        return lmeval_evaluate(model, task, sampled_examples)
    
class MatrixResultCollector:

    def __init__(self, filepath: str, layers: int = 32, heads: int = 32):
        self.filepath = filepath
        self.layers = layers
        self.heads = heads
        self.results_file = filepath
        self.mask_file = filepath.replace('.npy', '_mask.npy')
        
        if os.path.exists(self.results_file) and os.path.exists(self.mask_file):
            self.results = np.load(self.results_file)
            self.mask = np.load(self.mask_file)
            print(f"Loaded existing results: {self.get_completion_stats()}")
        else:
            self.results = np.zeros((layers, heads), dtype=float)
            self.mask = np.zeros((layers, heads), dtype=bool)
            self.save()
            print(f"Initialized new {layers}x{heads} matrix")
    
    def save(self):
        """Save current state to disk."""
        np.save(self.results_file, self.results)
        np.save(self.mask_file, self.mask)
    
    def get_pending_indices(self) -> List[Tuple[int, int]]:
        """Get list of (i, j) pairs that haven't been computed yet."""
        pending = np.argwhere(~self.mask)
        return [(int(i), int(j)) for i, j in pending]
    
    def get_completion_stats(self) -> str:
        """Return completion statistics as a string."""
        completed = np.sum(self.mask)
        total = self.layers * self.heads
        percentage = 100 * completed / total
        return f"{completed}/{total} ({percentage:.1f}%) complete"
    
    def set_result(self, i: int, j: int, value: float):
        """Set a single result and mark it as complete."""
        self.results[i, j] = value
        self.mask[i, j] = True
    
    def set_results_batch(self, indices: List[Tuple[int, int]], values: List[float]):
        """Set multiple results at once."""
        for (i, j), value in zip(indices, values):
            self.set_result(i, j, value)
    
    def is_complete(self) -> bool:
        """Check if all results have been computed."""
        return np.all(self.mask)
    
    def get_results(self) -> np.ndarray:
        """Get the current results matrix."""
        return self.results.copy()
    
def run_batch(collector: MatrixResultCollector, task, num_samples):
    """
    Run a batch of computations for pending results.
    
    Args:
        collector: MatrixResultCollector instance
        max_results: Maximum number of results to compute (None = compute all)
    """
    sampled_examples = get_samples(task, num_samples)

    pending = collector.get_pending_indices()
    
    if not pending:
        print("All results already computed!")
        return

    
    print(f"Task {task} with {num_samples} samples")
    print(f"Computing {len(pending)} results...")
    print(f"Status before: {collector.get_completion_stats()}")
    
    for idx, (i, j) in enumerate(pending):
        model, tokenizer = setup_and_ablate(args, i, [j])
        acc = task_eval(model, tokenizer, args.task, sampled_examples)
        print('acc: ', acc)
        collector.set_result(i, j, acc)
        
        # Save periodically (every 10 results) and at the end
        if (idx + 1) % 10 == 0 or (idx + 1) == len(pending):
            collector.save()
            print(f"------> Saved checkpoint: {idx + 1}/{len(pending)} results computed")
    
    print(f"Status after: {collector.get_completion_stats()}")
    
    if collector.is_complete():
        print("✓ All results computed!")


def main(args):
    model_sizes = {'llama3.1-8b-it': (32, 32), 'llama3.2-3b-it': (28, 24), 'llama3.2-1b-it': (16, 32)}
    n_layers, n_heads = model_sizes[args.model]
    model_name_str = args.model.replace('-', '_')
    if args.checkpoint:
        filename = f"results/{model_name_str}_{args.task}_{args.num_samples}_accs_{args.suffix}.npy"
        collector = MatrixResultCollector(filename, layers=n_layers, heads=n_heads)
        run_batch(collector, args.task, args.num_samples)
    
        # Get final results when complete
        if collector.is_complete():
            final_results = collector.get_results()
            print("\nFinal matrix shape:", final_results.shape)
            print("Final matrix sample (first 5x5):")
            print(final_results[:5, :5])
    else:
        if args.lmeval:
            num_samples = 100
            sampled_examples = get_samples(args.task, num_samples)
        else:
            sampled_examples = None
        start = time.time()
        all_accs = []
        for layer_id in range(n_layers):
            print("Layer ", layer_id)
            acc_lst = []
            for head_id in range(n_heads):
                model, tokenizer = setup_and_ablate(args, layer_id, [head_id])
                acc = task_eval(model, tokenizer, args.task, sampled_examples, args.num_samples)
                print(f"L{layer_id}H{head_id}: {acc:.4f}")
                acc_lst.append(acc)
            all_accs.append(acc_lst)
        all_accs = np.array(all_accs)
        print(f" Time: {time.time() - start}")
        
        np.save(f'results/{model_name_str}_{args.task}_{args.suffix}.npy', all_accs)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='llama3.1-8b-it')
    parser.add_argument('--access-token')
    parser.add_argument('--task', type=str, default='gsm8k')
    parser.add_argument('--num-samples', type=int, default=None)
    parser.add_argument('--single-run', action='store_true')
    parser.add_argument('--layerid', type=int, default=None, nargs='+')
    parser.add_argument('--headid', type=int, default=None, nargs='+')
    parser.add_argument('--extra-layers', type=int, default=None, nargs='+')
    parser.add_argument('--extra-heads', type=int, default=None, nargs='+')
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--checkpoint', action='store_true')
    args = parser.parse_args()
    args.lmeval = args.task not in NON_LMEVAL_TASKS
    print(args)
    if args.single_run:
        model, tokenizer = setup_and_ablate(args, layerid=args.layerid, headid=args.headid)
        if args.lmeval and args.num_samples is not None:
            sampled_examples = get_samples(args.task, num_samples=args.num_samples)
        else:
            sampled_examples = None
        acc = task_eval(model, tokenizer, args.task, sampled_examples, args.num_samples)
        print(round(acc, 4))
    else:
        main(args)
