## Evaluations

### Single runs to obtain accuracies
Model options: `llama3.1-8b-it`, `llama3.2-3b-it`, `llama3.2-1b-it`, `qwen2.5-7b-it`, `qwen2.5-3b-it`

Dataset options: `gsm8k`, `arithmetic`, `mbpp`, `humaneval_instruct`, `hellaswag`, `boolq`, `arc_challenge`, `mmlu`, `swearing`, `rhyming`

Get baseline accuracy of GSM8K dataset on Llama 3.1 8B model
```sh
python run_evals.py --single-run --model llama3.1-8b-it --task gsm8k
```

Accuracy with heads L16H21 and L15H13 ablated
```sh
python run_evals.py --single-run --model llama3.1-8b-it --task gsm8k --layerid 16 15 --headid 21 13
```

### Compressed sensing search

Perform compressed sensing with 100 masks, each of which has 0.02 sparsity. Include `--stratified` flag for stratified sampling in the measurement matrix, otherwise defaults to Bernoulli sampling.
```sh
python compressed_sensing.py --model llama3.1-8b-it --task gsm8k --nmasks 100 --sparsity 0.02 --num-samples 100 --stratified
```

### Greedy search

Greedy search via comprehensive ablation of all heads
```sh
python run_evals.py --model llama3.1-8b-it --task gsm8k --num-samples 100
```
Greedy search that additionally always knocks out L16H21 (i.e. for subsequent iterations of iterative greedy search) and does checkpointing for intermediate saving
```sh
python run_evals.py --model llama3.1-8b-it --task gsm8k --num-samples 100 --extra-layers 16 --extra-heads 21 --checkpoint
```

After performing greedy search, print out a list of the heads that resulted in the smallest accuracy on the target task:
```sh
python get_heads.py --filename {saved filepath here}
```





## Notes
- Added `unsafe_code: true` to arithmetic_1dc.yaml.