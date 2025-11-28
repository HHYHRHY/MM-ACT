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

This LIBERO is based on [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [OpenVLA](https://github.com/openvla/openvla.git). Thanks these great work.
