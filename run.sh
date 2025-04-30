export HF_HOME=/data/locus/project_data/project_data3/mingjies/huggingface
export HF_TOKEN=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
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TOKENIZERS_PARALLELISM=false

### default evaluation
CUDA_VISIBLE_DEVICES=0 python main_eval.py \
    --model llama3.1-8b-it \
    --arithmetic_type addition \
    --layer_id -1 \
    --attn_head_list -1 \
    --savedir results/

### zeroing target attn_head
CUDA_VISIBLE_DEVICES=0 python main_eval.py \
    --model llama3.1-8b-it \
    --arithmetic_type addition \
    --layer_id 16 \
    --attn_head_list 1 2 3 4 5 \
    --savedir results/

### zeroing random attn_head
CUDA_VISIBLE_DEVICES=0 python main_eval.py \
    --model llama3.1-8b-it \
    --arithmetic_type addition \
    --layer_id 16 \
    --attn_head_list -1 \
    --savedir results/
