#! /bin/bash

#SBATCH --job-name=eval_vmamba
#SBATCH --gres=gpu:2                # Request 1 GPUs total
#SBATCH --cpus-per-gpu=8            # Optional: 8 CPUs per GPU (so 24 total CPUs)
#SBATCH --mem=48G                   # Total memory requested for the job
#SBATCH --time=24:00:00             # ✅ Request 24 hours
#SBATCH --account=tolgalab
#SBATCH --output=/uu/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/Vim/output/resume0524/eval_slurm-%j.out
#SBATCH --error=/uu/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/Vim/output/resume0524/eval_slurm-%j.err
#SBATCH --partition=beasts,spartacus-tl,spartacus

# Activate conda environment
source ~/.bashrc
conda activate vmamba_dino

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 \
--use_env /uu/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/Vim/vim/main.py --eval --resume \
/uu/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/Vim/output/vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2/best_checkpoint.pth \
--model vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
--data-path /uu/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/Vim/data/streetlights \
--data_set STREET


# # python ./vim/main.py --eval --resume \
# CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 \
# --use_env /home/collab/u1368791/Vim/vim/main.py --eval --resume \
# ./output/vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2/best_checkpoint.pth \
# --model vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
# --data-path /home/collab/u1368791/Vim/data/streetlights --data_set STREET