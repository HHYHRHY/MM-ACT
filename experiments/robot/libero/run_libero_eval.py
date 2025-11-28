"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import AutoTokenizer
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torch.nn.functional as F
from typing import List, Optional, Union
import draccus
import numpy as np
import tqdm
from libero.libero import benchmark
import wandb
import argparse
import json
import time

# Append current directory so that interpreter can find experiments.robot
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.insert(0, project_root)
print(project_root)

from experiments.robot.robot_utils import DATE_TIME, set_seed_everywhere

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_action,
    quat2axisangle,
    save_rollout_video,
)

from deployment.mmact_deploy import MMACT_Deployment
from dataclasses import asdict


@dataclass
class LiberoConfig:
    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_dir: str = "MODEL_DIR"  # save directory of the model
    vq_model_path: str = "VQ_MODEL_PATH"  # path of the VQ-VAE model
    vocab_offset: int = 134656
    model_family: str = "MM-ACT"

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    resize_size: int = 256  # Resize size for images

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None  # Extra note to add in run ID for logging
    local_log_dir: str = "./results"  # Local directory for eval logs

    use_wandb: bool = False  # Whether to also log results in Weights & Biases
    wandb_project: str = "WANDB_PROJECT"  # Name of W&B project to log to (use default!)
    wandb_entity: str = "WANDB_ENTITY"  # Name of entity to log under

    seed: int = 7  # Random Seed (for reproducibility)

    # fmt: on
    #################################################################################################################
    # Experiment identifiers
    #################################################################################################################
    model_id: str = "default_model"  # model ID
    device_id: int = 0  # device ID (GPU)
    num_gpus: int = 1  # number of GPUs used
    run_id: int = 0  # run GPU ID
    test_id: int = 0  # test ID
    task_suite_name: str = "libero_object"  # task suite
    log_save_dir: str = ""
    rollout_save_dir: str = ""
    mp4_save_dir: str = ""
    timesteps: int = 1  # decoding timesteps
    preprocessing_max_seq_length: int = 1024
    training_chunk_size: int = 8
    action_dim: int = 7
    robot_type: str = "franka"
    is_save_video: bool = True


def eval_libero(
    cfg: LiberoConfig,
    model_id: str,
    run_id: int = 0,
    device_id: int = 0,
    num_gpus: int = 1,
    test_id: int = 1,
    task_suite_name: str = "libero_object",
    timesteps: int = 1,
) -> None:
    assert cfg.model_dir is not None, "cfg.model_dir must not be None"
    assert model_id is not None, "model_id must not be None"
    assert cfg.vq_model_path is not None, "cfg.vq_model_path must not be None"

    # Set seed
    set_seed_everywhere(cfg.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    torch.cuda.set_device(device_id)

    # Load model
    model_path = os.path.join(cfg.model_dir, model_id)
    mmact = MMACT_Deployment(
        model_path=model_path,
        vq_model_path=cfg.vq_model_path,
        vocab_offset=cfg.vocab_offset,
        device=f"cuda:{device_id}",
        timesteps=cfg.timesteps,
        preprocessing_max_seq_length=cfg.preprocessing_max_seq_length,
        training_chunk_size=cfg.training_chunk_size,
        action_dim=cfg.action_dim,
        robot_type=cfg.robot_type,
    )
    model_id = model_id.replace(
        "/", ""
    )  # because some of the model id contains "/checkpoint_i"
    # Load libero environment
    run_log_name = (
        f"EVAL-{model_id}--{task_suite_name}--{DATE_TIME}--{test_id}--{run_id}"
    )
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    rollout_save_dir = os.path.join(cfg.local_log_dir, "rollouts")
    os.makedirs(rollout_save_dir, exist_ok=True)
    mp4_save_dir = os.path.join(
        rollout_save_dir, model_id, task_suite_name, "agentview", f"test_{test_id}"
    )
    os.makedirs(mp4_save_dir, exist_ok=True)
    log_save_dir = os.path.join(cfg.local_log_dir, "logs", model_id, task_suite_name)
    os.makedirs(log_save_dir, exist_ok=True)
    local_log_filepath = os.path.join(log_save_dir, run_log_name + ".txt")
    log_file = open(local_log_filepath, "w")
    local_config_filepath = os.path.join(log_save_dir, run_log_name + ".json")

    cfg.model_id = model_id
    cfg.run_id = run_id
    cfg.device_id = device_id
    cfg.num_gpus = num_gpus
    cfg.test_id = test_id
    cfg.task_suite_name = task_suite_name
    cfg.log_save_dir = log_save_dir
    cfg.rollout_save_dir = rollout_save_dir
    cfg.timesteps = timesteps
    cfg.mp4_save_dir = mp4_save_dir

    with open(local_config_filepath, "w") as f:
        json.dump(asdict(cfg), f)
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {task_suite_name}")
    log_file.write(f"Task suite: {task_suite_name}\n")

    # Get expected image dimensions
    resize_size = cfg.resize_size

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, resolution=256)

        # Start episodes
        task_episodes, task_successes = 0, 0
        all_indices = list(range(cfg.num_trials_per_task))
        assigned_indices = [i for i in all_indices if i % num_gpus == run_id]
        log_file.write(f"Episode indices: {assigned_indices}\n")

        for episode_idx in tqdm.tqdm(assigned_indices):
            print(f"\nTask: {task_description}")
            print(f"episode: {episode_idx}")
            log_file.write(f"\nTask: {task_description}\n")
            log_file.write(f"episode: {episode_idx}\n")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_agent_images = []
            replay_wrist_images = []
            if task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif task_suite_name == "libero_10":
                max_steps = 520  # 520  # longest training demo has 505 steps
            elif task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            while t < max_steps + cfg.num_steps_wait:
                finish_flag = 0
                # try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < cfg.num_steps_wait:
                    obs, reward, done, info = env.step(
                        get_libero_dummy_action(cfg.model_family)
                    )
                    t += 1
                    continue

                # Get preprocessed image
                agent_img, wrist_img, agent_img_tensor, wrist_img_tensor = (
                    get_libero_image(obs, resize_size)
                )

                # Save preprocessed image for replay video
                replay_agent_images.append(agent_img)
                replay_wrist_images.append(wrist_img)
                images_tensor = [agent_img_tensor, wrist_img_tensor]
                text_task = str(task_description)
                state_tensor = get_libero_action(obs)
                state_tensor[3:6] = torch.tensor(
                    [x / 5.0 for x in state_tensor[3:6]]
                )  # quickly fix bug in state

                flat_prev_actions_tensors = torch.tensor([])
                action_chunk, token_ids = mmact.get_actions(
                    inputs=(
                        images_tensor,
                        text_task,
                        state_tensor,
                        flat_prev_actions_tensors,
                    ),
                )
                action_chunk = action_chunk[:8].to(
                    "cpu"
                )  # magic number, need to be refined
                for i, action in enumerate(action_chunk):
                    # Execute action in environment
                    t += 1
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        finish_flag = 1
                        break
                if finish_flag:
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            if cfg.is_save_video:
                save_rollout_video(
                    rollout_save_dir,
                    replay_agent_images,
                    model_id,
                    task_suite_name,
                    test_id,
                    task_id,
                    episode_idx,
                    success=done,
                    task_description=task_description,
                    type="agentview",
                    log_file=log_file,
                )
                save_rollout_video(
                    rollout_save_dir,
                    replay_wrist_images,
                    model_id,
                    task_suite_name,
                    test_id,
                    task_id,
                    episode_idx,
                    success=done,
                    task_description=task_description,
                    type="wristview",
                    log_file=log_file,
                )

            # Log current results
            print(f"Success: {done}")
            print(f"# task episodes completed: {task_episodes}")
            print(
                f"# task successes: {task_successes} ({task_successes / task_episodes * 100:.1f}%)"
            )
            print(f"# episodes completed so far: {total_episodes}")
            print(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)"
            )
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# task episodes completed: {task_episodes}\n")
            log_file.write(
                f"# task successes: {task_successes} ({task_successes / task_episodes * 100:.1f}%)\n"
            )
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n"
            )
            log_file.flush()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu_id",
        type=int,
        required=True,
        help="GPU ID to use (each process uses a separate GPU)",
    )
    parser.add_argument(
        "--run_id",
        type=int,
        required=True,
        help="Run GPU ID to use (each process uses a separate run GPU ID)",
    )
    parser.add_argument(
        "--num_gpus", type=int, required=True, help="Number of GPUs to use"
    )
    parser.add_argument("--test_id", type=int, required=True, help="Test ID to use")
    parser.add_argument("--model_id", type=str, required=True, help="Model ID to use")
    parser.add_argument(
        "--task_suite_name",
        type=str,
        required=True,
        default="libero_object",
        choices=[
            "libero_spatial",
            "libero_object",
            "libero_goal",
            "libero_10",
            "libero_90",
        ],
        help="Task suite to use",
    )
    parser.add_argument(
        "--timesteps", type=int, required=True, help="Decoding timesteps to use"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = LiberoConfig()
    eval_libero(
        cfg,
        args.model_id,
        args.run_id,
        args.gpu_id,
        args.num_gpus,
        args.test_id,
        args.task_suite_name,
        args.timesteps,
    )


if __name__ == "__main__":
    main()
