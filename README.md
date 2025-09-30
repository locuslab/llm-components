## Commands

Baseline accuracy on a given model:
```sh
python main_eval.py --model llama3.1-8b-it --dataset twodig --single_run
```

Accuracy with a head ablated:
```sh
python main_eval.py --model llama3.1-8b-it --dataset twodig --single_run --layer_id_list 15 --attn_head_list 13 
```

Accuracy with the complement of a head ablated:
```sh
python main_eval.py --model llama3.1-8b-it --dataset twodig --single_run --layer_id_list 15 --attn_head_list 13 --complement
```

Collect accuracies for ablating each head in turn and save results:
```sh
python main_eval.py --model llama3.1-8b-it --dataset twodig 
```

Default is to do zero ablation, add `--mean_ablation` flag to do mean_ablation instead or `--patching` for activation patching.

Examples for evaluating on LM Eval Harness are included in `lm_eval.sh`.
