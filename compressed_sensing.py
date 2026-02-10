import os
import time
import torch
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple
from tqdm import tqdm
import argparse
import lm_eval
from lm_eval.models.huggingface import HFLM

from load_model import load_llm_hf
from modify_model import zero_ablate_attn_head
from run_evals import task_eval, get_model_dir
from evals import get_samples

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class HeadImportanceIdentifier:
    def __init__(self, args, device: str = "cuda"):
        self.device = device
        self.args = args
        model, _ = self.reset_model()
        
        self.n_layers = model.model.config.num_hidden_layers
        self.n_heads = model.model.config.num_attention_heads
        self.total_heads = self.n_layers * self.n_heads
        self.num_samples = args.num_samples
        self.sampled_examples = get_samples(args.task, num_samples=args.num_samples) if args.lmeval else None 

        self.stratified = args.stratified
    
    def reset_model(self):
        if self.args.lmeval:
            model_dir = get_model_dir(self.args.model)
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
        else:
            model, tokenizer = load_llm_hf(self.args)
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    def get_lmeval_model(self):
        model_dir = get_model_dir(self.args.model)
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
        return model, tokenizer

    
    def generate_random_masks(self, n_masks, sparsity):
        if self.stratified:
            masks = np.zeros((n_masks, self.total_heads))
            heads_per_mask = int(sparsity * self.total_heads)
            appearance_counts = np.zeros(self.total_heads)

            for i in range(n_masks):
                # Select heads with fewest appearances so far
                scores = -appearance_counts + np.random.randn(self.total_heads) * 0.01
                selected = np.argsort(scores)[-heads_per_mask:]
                
                masks[i, selected] = 1
                appearance_counts[selected] += 1
        else:
            masks = np.random.binomial(1, sparsity, (n_masks, self.total_heads))
        return masks # shape n_masks, total_heads

    def apply_head_mask(self, mask, model, lmeval):
        mask_2d = mask.reshape(self.n_layers, self.n_heads)
        for layer_idx in range(self.n_layers):
            if lmeval:
                layer = model.model.model.layers[layer_idx]
            else:
                layer = model.model.layers[layer_idx]
            head_lst =  np.where(mask_2d[layer_idx])[0].tolist()
            zero_ablate_attn_head(layer, head_lst)

    def evaluate_with_mask(self, mask, task, lmeval):
        model, tokenizer = self.reset_model()
        self.apply_head_mask(mask, model, lmeval)
        acc = task_eval(model, tokenizer, task, self.sampled_examples, self.num_samples)
        del model, tokenizer
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        import gc; gc.collect()
        return acc
        
    
    def identify_important_heads(self, task, lmeval, n_masks, sparsity, alpha):

        print(f"Generating {n_masks} random masks...")
        masks = self.generate_random_masks(n_masks, sparsity)
        
        print(f"Evaluating model with each mask...")
        accuracies = []
        for i, mask in enumerate(tqdm(masks)):
            acc = self.evaluate_with_mask(mask, task, lmeval)
            accuracies.append(acc)
            if (i + 1) % 10 == 0:
                print(f"  Mask {i+1}/{n_masks}: accuracy = {acc*100:.3f}")
        
        accuracies = np.array(accuracies)
        
        print(f"\nFitting regression model...")
        print(f"Accuracy range: [{accuracies.min():.3f}, {accuracies.max():.3f}]")
        scaler = StandardScaler()
        accuracies_scaled = scaler.fit_transform(accuracies.reshape(-1, 1)).ravel()
        model = Lasso(alpha=alpha, max_iter=5000)

        model.fit(masks, accuracies_scaled)
        head_importance = model.coef_
        top_heads = self.get_top_heads(head_importance)
        bottom_heads = self.get_bottom_heads(head_importance)
        print(f"R² score: {model.score(masks, accuracies_scaled):.3f}")

    
    def thompson_sampling(self, task, lmeval, n_iterations, sparsity):
        alphas = np.ones(self.total_heads)
        betas = np.ones(self.total_heads)

        for iteration in range(n_iterations):
            # Sample from Beta distributions
            samples = np.random.beta(alphas, betas)
            
            # Select top heads based on samples
            n_select = int(self.total_heads * sparsity)
            selected = np.argsort(samples)[-n_select:]
            
            mask = np.zeros(self.total_heads)
            mask[selected] = 1
            
            # Evaluate
            acc = self.evaluate_with_mask(mask, task, lmeval)
            
            # Update Beta distributions
            # Heads that were kept in a high-accuracy mask get alpha boost
            # Heads that were ablated get beta boost
            reward = acc #1-acc # Minimize accuracy
            alphas[selected] += reward
            betas[selected] += (1 - reward)
            
            ablated = np.where(mask == 0)[0]
            alphas[ablated] += (1 - reward) * 0.5
            betas[ablated] += reward * 0.5
            
            if (iteration + 1) % 1 == 0:
                print(f"Iteration {iteration + 1}/{n_iterations}: accuracy = {acc:.3f}")
                # Show top heads
                expectations = alphas / (alphas + betas)
                top_idx = np.argsort(expectations)[-5:][::-1]
                print("  Top-5 heads (layer, head, prob):")
                for idx in top_idx:
                    layer = idx // self.n_heads
                    head = idx % self.n_heads
                    print(f"    ({layer:2d}, {head:2d}): {expectations[idx]:.4f}")
        
        # Final importance = expected value of Beta distribution
        head_importance = alphas / (alphas + betas)
        results = {
            'alphas': alphas,
            'betas': betas,
            'expectations': head_importance
        }
        
        return head_importance, results

    
    def get_top_heads(
        self,
        head_importance: np.ndarray,
        top_k: int = 20
    ) -> List[Tuple[int, int, float]]:
        top_indices = np.argsort(head_importance)[-top_k:][::-1]
        top_heads = []
        for idx in top_indices:
            layer_idx = idx // self.n_heads
            head_idx = idx % self.n_heads
            importance = head_importance[idx]
            top_heads.append((layer_idx, head_idx, importance))
        
        print("\nTop 20 most important heads:")
        print("Rank | Layer | Head | Importance")
        print("-" * 40)
        layers = []
        heads = []
        for rank, (layer, head, importance) in enumerate(top_heads, 1):
            print(f"{rank:4d} | {layer:5d} | {head:4d} | {importance:.6f}")
        print([(l, h) for l, h in zip(layers, heads)])
        return top_heads

    def get_bottom_heads(
        self,
        head_importance: np.ndarray,
        top_k: int = 20
    ) -> List[Tuple[int, int, float]]:
        top_indices = np.argsort(head_importance)[:top_k]
        bottom_heads = []
        for idx in top_indices:
            layer_idx = idx // self.n_heads
            head_idx = idx % self.n_heads
            importance = head_importance[idx]
            bottom_heads.append((layer_idx, head_idx, importance))
        
        print("\nBottom 10 most important heads:")
        print("Rank | Layer | Head | Importance")
        print("-" * 40)
        layers = []
        heads = []
        for rank, (layer, head, importance) in enumerate(bottom_heads, 1):
            print(f"{rank:4d} | {layer:5d} | {head:4d} | {importance:.6f}")
            layers.append(layer)
            heads.append(head)
        print([(l, h) for l, h in zip(layers, heads)])
        return bottom_heads
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='llama3.1-8b-it')
    parser.add_argument('--access-token')
    parser.add_argument('--task', type=str, default='gsm8k')
    parser.add_argument('--num-samples', type=int, default=20)
    parser.add_argument('--nmasks', type=int, default=100)
    parser.add_argument('--sparsity', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--strategy', type=str, default='random')
    parser.add_argument('--thompson_iters', type=int, default=50)
    parser.add_argument('--stratified', action='store_true')
    args = parser.parse_args()
    args.lmeval = args.task not in ['swearing', 'rhyming', 'counting']
    print(args)
    identifier = HeadImportanceIdentifier(args)
    start = time.time()
    if args.strategy == 'random':
        identifier.identify_important_heads(
            task=args.task, 
            lmeval=args.lmeval, 
            n_masks=args.nmasks, 
            sparsity=args.sparsity, 
            alpha=args.alpha, 
        )
    elif args.strategy == 'thompson':
        head_importance, results = identifier.thompson_sampling(
            task=args.task, 
            lmeval=args.lmeval, 
            n_iterations=args.thompson_iters, 
            sparsity=args.sparsity, 
        )
    print(f"Total time: {((time.time() - start)/60):4f} minutes")
