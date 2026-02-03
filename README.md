## Evaluations

Get baseline accuracy of GSM8K dataset on Llama 3.1 8B model
```sh
python run_evals.py --single-run --model llama3.1-8b-it --task gsm8k
```

Accuracy with heads L16H21 and L15H13 ablated
```sh
python run_evals.py --single-run --model llama3.1-8b-it --task gsm8k --layerid 16 15 --headid 21 13
```

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