#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=deepacf
#SBATCH --cpus-per-task=1
#SBATCH --output=run-out.%j
#SBATCH --error=run-err.%j
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpus

source ../../../transformers_modules.sh
source ../../transformers_env/bin/activate

module list
export CUDA_VISIBLE_DEVICES="0"

export MASTER_PORT=12340
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

#srun python vistrans.py --gpus 1 \
#	--data_dir /p/scratch/deepacf/kiste/patnala1/eurosat/2750 \
#	--img_size 64 \
#	--num_classes 10

srun python swin_transformer.py --gpus 1 \
	--data_dir /p/scratch/deepacf/kiste/patnala1/eurosat/2750 \
	--img_size 64 \
	--num_classes 10 \
	--window_size 4 \
	--patch_size 4





