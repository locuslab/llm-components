### May need to be moved to lm-evaluation-harness dir depending on setup
#!/bin/bash
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=100GB
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
##SBATCH --exclude=babel-0-37
##SBATCH --partition=preempt

export HF_TOKEN=""
export TOKENIZERS_PARALLELISM="false"



#### Baseline
echo "----- Baseline -----"
python lm_eval.py --task $1 

#### Single head
echo "----- GSM8k top head: Zero Ablate L2H4 -----"
python lm_eval.py --zero_ablate --layerid 2 --attn_head_list 4 --task $1

#### Top Math Heads
echo "----- Zero Ablate L15H13, L16H21, L15H14 -----"
python lm_eval.py --zero_ablate --layer_id_list 15 16 15 --attn_head_list 13 21 14 --task $1
