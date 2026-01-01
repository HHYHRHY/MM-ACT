# LIBERO Evaluation

## LIBERO Setup

Clone and install the [LIBERO repo](https://github.com/Lifelong-Robot-Learning/LIBERO):

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
```

Additionally, install other required packages:

```bash
cd MM-ACT
pip install -r experiments/robot/libero/libero_requirements.txt
```

## Evaluating MM-ACT on LIBERO

```bash
python experiments/robot/libero/run_libero_eval.py \
    --gpu_id $GPU_ID \
    --run_id $RUN_ID \
    --num_gpus $NUM_GPUS \
    --test_id $TEST_ID \
    --model_id $MODEL_ID \
    --task_suite_name $TASK_SUITE_NAME \
    --timesteps $TIMESTEPS
```

**Parameters:**

- `--gpu_id` (int, required): GPU ID to use. Each process should use a separate GPU.
- `--run_id` (int, required): Run ID for distributed evaluation. Each process uses a separate run ID to distribute tasks across multiple GPUs. Tasks are assigned based on `episode_idx % num_gpus == run_id`.
- `--num_gpus` (int, required): Number of GPUs used for parallel evaluation.
- `--test_id` (int, required): Test ID to identify different evaluation runs.
- `--model_id` (str, required): Model ID/name to load from the model directory (specified in `LiberoConfig.model_dir`).
- `--task_suite_name` (str, required): LIBERO task suite to evaluate on. Options: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90`.
- `--timesteps` (int, required): Number of decoding timesteps for action generation.

To run the evaluation on a single GPU, use the following command:

```bash
python experiments/robot/libero/run_libero_eval.py \
    --gpu_id 0 \
    --run_id 0 \
    --num_gpus 1 \
    --test_id 1 \
    --model_id default_model \
    --task_suite_name libero_object \
    --timesteps 1
```

## Acknowledgments

This LIBERO evaluation is based on [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [OpenVLA](https://github.com/openvla/openvla.git). Thanks these great work.

# RoboTwin Evaluation

## RoboTwin Setup

Clone and install [RoboTwin repo](https://github.com/RoboTwin-Platform/RoboTwin.git).
Then copy the MM-ACT RoboTwin evaluation scripts and model/training code from this repository into the corresponding locations inside the `RoboTwin` repository.

```bash
# Assume the directory layout is:
# <your_path>/MM-ACT
# <your_path>/RoboTwin

cd <your_path>

# 1) Copy MM-ACT RoboTwin evaluation scripts into RoboTwin policy directory
cp -r MM-ACT/experiments/robot/robotwin/* RoboTwin/policy/MM-ACT/

# 2) Copy MM-ACT model and training code into RoboTwin policy directory
cp -r MM-ACT/models RoboTwin/policy/MM-ACT/
cp -r MM-ACT/training RoboTwin/policy/MM-ACT/
```

After copying, the MM-ACT-related directory structure inside the `RoboTwin` repository will look approximately as follows (only relevant parts are shown):

```text
RoboTwin/
  policy/
    MM-ACT/
      __init__.py
      eval.sh
      deploy_policy.py
      deploy_policy.yml
      models/
        ...
      training/
        ...
```
Both our model and baselines in our paper are trained using data collected from L515 cameras (including both head and wrist cameras). To test with our models, please modify the relevant parameters in RoboTwin configuration file (task_config/demo_randomized.yml).

## Install evaluation environment
You can directly install MM-ACT's requirements in a RoboTwin environment.
```bash
# After installing a RoboTwin conda environment, run the installation command of MM-ACT inside RoboTwin
conda activate RoboTwin
cd MM-ACT
pip install -r requirements.txt
```
If you run into segmentation faults during evaluation, please check if the numpy version is too high. Downgrading numpy to numpy==1.26.0 should solve this problem.
```bash
pip install numpy==1.26.0
```

## Acknowledgments

This RoboTwin evaluation is based on [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin.git). Thanks these great work.
