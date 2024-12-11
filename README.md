## Commands

collect mean ablation for one attention head:
```sh
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model llama3.1-8b-it \
    --head_str L0H0 \
    --collect_mean \
    --savedir results/llama3.1-8b-it/mean_ablate/
```
After this step, "mean_outputs_100.pt" will be stored for the mean outputs of L0H0 attn head over 100 sequences.

Then we use this ablation to get the token-wise loss for mean ablation one attention head, e.g., L0H0 (layer 0 attn head 0).
```sh
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model llama3.1-8b-it \
    --head_str L0H0 \
    --savedir results/llama3.1-8b-it/mean_ablate/
```

Note that zero ablation does not require collecting any statistics, where we can simply zero out the outputs of one attn head, can use `mp.zero_ablate_llama3_attn_head(layers[layer_id], attn_id)` for zero ablation