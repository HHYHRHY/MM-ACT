# MM-ACT: Learn from Multimodal Parallel Generation to Act

[![arXiv](https://img.shields.io/badge/arXiv-Paper-red.svg)](https://arxiv.org/abs/2512.00975)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97-Model-yellow)](https://huggingface.co/hhyhrhy/MM-ACT-Model)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97-Dataset-blue)](https://huggingface.co/datasets/hhyhrhy/MM-ACT-data)

<br>

<div align="center">
  <img src="assets/MM-ACT.png" width="80%" alt="MM-ACT Arch"/>
</div>

<br>

**MM-ACT** is a unified model that integrates text, image, and action into a shared token space, performing generation across all three modalities.
It adopts re-mask parallel decoding strategy for text and image generation, and employs one-step parallel decoding strategy for action generation to improve efficiency.

This repository contains:

- Training pipelines and deployment scripts for one-step parallel decoding and re-mask parallel decoding strategies across three modalities: action, text, and image.
- Scripts for evaluation on LIBERO and Robotwin, as well as the data collection pipeline used for task planning annotation on Robotwin.

## 🛠️ Installation

### 1. Clone Repo and Environment Setup

```bash
git clone https://github.com/HHYHRHY/MM-ACT.git
cd MM-ACT

# Create environment
conda create -n mmact python=3.13
conda activate mmact

# Install requirements
pip install -r requirements.txt
```

### 2. Dataset Preparation

- **LIBERO**

  We utilize LIBERO datasets from [Huggingface_LeRobot](https://huggingface.co/lerobot), and uses LeRobot datasets for loading robot data.
  Please download [LIBERO-Object](https://huggingface.co/datasets/lerobot/libero_object_image),
  [LIBERO-Spatial](https://huggingface.co/datasets/lerobot/libero_spatial_image),[LIBERO-Goal](https://huggingface.co/datasets/lerobot/libero_goal_image) and
  [LIBERO-10](https://huggingface.co/datasets/lerobot/libero_10_image). For LIBERO-10, we also provide our task planning datasets in [LIBERO-10-task](https://huggingface.co/datasets/hhyhrhy/MM-ACT-data/tree/main/LIBERO).

- **RoboTwin**

  For RoboTwin datasets, we utilize a dataset sampling pipeline that includes task planning generation. You can download our [datasets](https://huggingface.co/datasets/hhyhrhy/MM-ACT-data/tree/main/RoboTwin)
  or collect your own datasets with our pipeline in [Robotwin_subtask](https://github.com/RoboTwin-Platform/RoboTwin/tree/Subtask_info). This branch includes updates to original RoboTwin data collection pipeline to support our subtask text annotations. The collection usage is identical to the main branch. Please report any bugs or questions of text annotations in MM-ACT's issue.

### 3. Model Weight Preparation

Download the base model weights from MMaDA: [MMaDA-8B-Base](https://huggingface.co/Gen-Verse/MMaDA-8B-Base) and expand the original model's action codebook (we use 2048):

```bash
cp models/configuration_llada.py models/modeling_llada.py ${origin_model_path}
python model_utils/resize_model_vocab.py --model ${origin_model_path} --out ${output_model_path} --num_new ${action_codebook_size}
```

Besides, please download the image quantizer weight from [showlab/magvitv2](https://huggingface.co/showlab/magvitv2/tree/main) 
and update your local weight path (e.g., ```""/xxx/magvitv2""```) in ```vq_model_name``` (training configs) and ```vq_model_path``` (experiments and deployment configs).

## 🚀 Training

We provide training pipelines for both LIBERO and RoboTwin. You can refer to the explanations of the configuration settings in [configs/README.md](configs/README.md).
Single-node training can be launched using accelerate:

```bash
accelerate launch \
  --config_file accelerate_configs/1_node_8_gpus_deepspeed_zero2.yaml \
  --main_process_port 8888 \
  training/{your_training_script}.py \
  config=configs/{your_training_config}.yaml
```

Multi-node training can be referenced from the script in [shell/training.sh](shell/training.sh) and adapted to the launch commands of your cluster.
For **LIBERO**, We provide three specific pipelines:

1.Text-Only Training: [training/train_mmact_libero_mmu.py](training/train_mmact_libero_mmu.py), which used in LIBERO-10 stage1 training.

2.Action-Only Training: [training/train_mmact_libero_action.py](training/train_mmact_libero_action.py) wich used in all of LIBERO benchmark for action training.

3.Mixed (Text & Action) Training: [training/train_mmact_libero_mix.py](training/train_mmact_libero_mix.py), which used in LIBERO-10 stage2 training.

For **RoboTwin**, we provide a unified mixed-modality pipeline in [training/train_mmact_robotwin_mix.py](training/train_mmact_robotwin_mix.py).
You can achieve arbitrary modality combinations training by adjusting parameters in configs.

For real robot, You can first convert your real-robot data into the LeRobot format.
Then refer to the [training/train_mmact_libero_action.py](training/train_mmact_libero_action.py) script to conduct real-robot data training.

## ⚡ Evaluation & Deployment

Our trained model weights can be found at: [MM-ACT-weights](https://huggingface.co/hhyhrhy/MM-ACT-Model).

For **LIBERO** and **RoboTwin** evalutation, please refer to [experiments/README.md](experiments/README.md) for detailed instructions.

For real-world deployment, please refer to the script provided at: [deployment/mmact_deploy.py](deployment/mmact_deploy.py)

## 🎥 Real-world Experiments (Video Demo)

https://private-user-images.githubusercontent.com/91517920/520774696-02a3bf40-f1ae-4f52-9562-a3fc2e9a1477.mp4

## Citation

If you find our work helpful, please cite us:

```bibtex
@article{liang2025mm,
  title={MM-ACT: Learn from Multimodal Parallel Generation to Act},
  author={Liang, Haotian and Chen, Xinyi and Wang, Bin and Chen, Mingkang and Liu, Yitian and Zhang, Yuhao and Chen, Zanxin and Yang, Tianshuo and Chen, Yilun and Pang, Jiangmiao and others},
  journal={arXiv preprint arXiv:2512.00975},
  year={2025}
}
```

## Acknowledgments

This work is based on [MMaDA](https://github.com/Gen-Verse/MMaDA), [RoboTwin](https://github.com/robotwin-Platform/RoboTwin),
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [LeRobot](https://github.com/huggingface/lerobot), [OpenVLA](https://github.com/openvla/openvla.git). Thanks these great work.
