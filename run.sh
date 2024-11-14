export HF_HOME=/data/locus/project_data/project_data3/mingjies/huggingface
export HF_TOKEN=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
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=0 python main.py \
    --model llama3.1-8b-it \
    --savedir results/model/llama3.1-8b-it