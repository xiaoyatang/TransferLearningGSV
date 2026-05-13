#! /bin/bash

#SBATCH --job-name=ft_vmamba_30eps
#SBATCH --gres=gpu:2              # Request 2 GPUs total
#SBATCH --cpus-per-gpu=8            # Optional: 8 CPUs per GPU (so 24 total CPUs)
#SBATCH --mem=24G                   # Total memory requested for the job
#SBATCH --time=24:00:00             # ✅ Request 24 hours
#SBATCH --account=tolgalab
#SBATCH --output=/uufs/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/Vim/output/resume0524/slurm-%j.out
#SBATCH --error=/uufs/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/Vim/output/resume0524/slurm-%j.err
#SBATCH --partition=gods

# Activate conda environment
source ~/.bashrc
conda activate vmamba_dino_cuda11

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 \
--use_env /uufs/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/Vim/vim/main.py --model \
vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2 \
--batch-size 32 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8 \
--num_workers 25 \
--data-path /uufs/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/Vim/data/streetlights \
--output_dir \
/uufs/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/unsupervised_pretrained_models_eval/streetlight18k/output \
--epochs 30 --finetune /uufs/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/DINO/vmambab_gsv_IN1Kpretrained/vim_b_midclstok_81p9acc.pth --no_amp

