
## Envs. for Pretraining

- NVIDIA GPUs:
  - Python 3.10.13

    - `conda create -n your_env_name python=3.10.13`

  - torch 2.1.1 + cu118
    - `pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118`

- AMD GPUs:
  - A [Docker image](https://hub.docker.com/r/rocm/pytorch/tags) is recommended, such as `rocm/pytorch:rocm6.2_ubuntu20.04_py3.9_pytorch_release_2.1.2`. For step-by-step instructions, please refer to [Vision Mamba on AMD GPU with ROCm](https://rocm.blogs.amd.com/artificial-intelligence/vision-mamba/README.html#vision-mamba).


- Requirements: vim_requirements.txt
  - `pip install -r vim/vim_requirements.txt`

- Install ``causal_conv1d`` and ``mamba``
  - `pip install -e causal_conv1d>=1.1.0`
  - `pip install -e mamba-1p1p1`
  
  


## Train Your Vim

`bash vim/scripts/pt-vim-t.sh`

## Train Your Vim at Finer Granularity
`bash vim/scripts/ft-vim-t.sh`

## Model Weights

| Model | #param. | Top-1 Acc. | Top-5 Acc. | Hugginface Repo |
|:------------------------------------------------------------------:|:-------------:|:----------:|:----------:|:----------:|
| [Vim-tiny](https://huggingface.co/hustvl/Vim-tiny-midclstok)    |       7M       |   76.1   | 93.0 | https://huggingface.co/hustvl/Vim-tiny-midclstok |
| [Vim-tiny<sup>+</sup>](https://huggingface.co/hustvl/Vim-tiny-midclstok)    |       7M       |   78.3   | 94.2 | https://huggingface.co/hustvl/Vim-tiny-midclstok |
| [Vim-small](https://huggingface.co/hustvl/Vim-small-midclstok)    |       26M       |   80.5   | 95.1 | https://huggingface.co/hustvl/Vim-small-midclstok |
| [Vim-small<sup>+</sup>](https://huggingface.co/hustvl/Vim-small-midclstok)    |       26M       |   81.6   | 95.4 | https://huggingface.co/hustvl/Vim-small-midclstok |
| [Vim-base](https://huggingface.co/hustvl/Vim-base-midclstok)    |       98M       |   81.9   | 95.8 | https://huggingface.co/hustvl/Vim-base-midclstok |

**Notes:**
- <sup>+</sup> means that we finetune at finer granularity with short schedule.
## Evaluation on Provided Weights
To evaluate `Vim-Ti` on ImageNet-1K, run:
```bash
python main.py --eval --resume /path/to/ckpt --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-path /path/to/imagenet
```
## Acknowledgement :heart:
This project is based on Vim([paper](https://icml.cc/virtual/2024/poster/33768)), (Mamba ([paper](https://arxiv.org/abs/2312.00752), [code](https://github.com/state-spaces/mamba)), Causal-Conv1d ([code](https://github.com/Dao-AILab/causal-conv1d)), DeiT ([paper](https://arxiv.org/abs/2012.12877), [code](https://github.com/facebookresearch/deit)). Thanks for their wonderful works.

## Citation
If you find Vim is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.

```bibtex
 @inproceedings{vim,
  title={Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model},
  author={Zhu, Lianghui and Liao, Bencheng and Zhang, Qian and Wang, Xinlong and Liu, Wenyu and Wang, Xinggang},
  booktitle={Forty-first International Conference on Machine Learning}
}
```
