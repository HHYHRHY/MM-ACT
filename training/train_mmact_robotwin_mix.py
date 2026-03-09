import os
import sys
from pathlib import Path
import torch
import logging
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from accelerate import Accelerator
from accelerate.utils import DistributedType, set_seed
from accelerate.logging import get_logger
from transformers import AutoTokenizer

from tqdm import tqdm
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.utils import (
    get_config,
    mask_or_random_replace_action_tokens,
    mask_or_random_replace_tokens,
)
from training.prompting_utils import UniversalPrompting
from training.robotwin_dataset import EpisodesHDF5Dataset
from models import MMACTModelLM, MAGVITv2, get_mask_schedule
from models.lr_schedulers import get_scheduler


def setup_logger(log_file: Path, is_main_process: bool) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)

    for h in list(logger.handlers):
        logger.removeHandler(h)

    if is_main_process:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


def main():
    config = get_config()
    output_dir = Path(config.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir = output_dir / "logs"
    logging_dir.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.get(
            "gradient_accumulation_steps", 1
        ),
        mixed_precision=config.training.get("mixed_precision", "bf16"),
        log_with="tensorboard",
        project_dir=str(logging_dir),
        split_batches=True,
    )
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config[
            "train_micro_batch_size_per_gpu"
        ] = config.training.batch_size
    accelerator.init_trackers(
        project_name="mmact_train",
        config={
            "batch_size": config.training.batch_size,
            "lr": config.optimizer.params.learning_rate,
            "epochs": config.training.num_epochs,
        },
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.mmact.pretrained_model_path,
        padding_side="left",
        local_files_only=True,
    )
    model = MMACTModelLM.from_pretrained(
        config.model.mmact.pretrained_model_path, torch_dtype=torch.bfloat16
    )

    model.to(accelerator.device)
    action_vocab_size = model.config.action_vocab_size
    max_action_prompt_len = (
        config.dataset.preprocessing.max_seq_length
        - config.training.chunk_size * config.training.action_dim
        - 2
    )
    # 256 is the default num_vq_tokens per image in MMada; 2 for <|soi|> and <|eoi|>
    max_image_seq_length = config.dataset.preprocessing.max_seq_length - 256 - 2
    uni_prompting = UniversalPrompting(
        tokenizer,
        max_text_len=config.dataset.preprocessing.max_seq_length,
        max_action_prompt_len=max_action_prompt_len,
        special_tokens=(
            "<|soi|>",
            "<|eoi|>",
            "<|sov|>",
            "<|eov|>",
            "<|t2i|>",
            "<|mmu|>",
            "<|t2v|>",
            "<|v2v|>",
            "<|lvg|>",
            "<|mm2a|>",
            "<|soa|>",
            "<|eoa|>",
            "<|7dim|>",
            "<|14dim|>",
            "<|sostate|>",
            "<|eostate|>",
        ),
        ignore_id=-100,
        cond_dropout_prob=0.0,
        use_reserved_token=True,
        action_vocab_size=action_vocab_size,
    )

    # load VQ model for image tokenization
    vq_cfg = config.model.vq_model
    if vq_cfg.type == "magvitv2":
        vq_model = MAGVITv2.from_pretrained(vq_cfg.vq_model_name).to(accelerator.device)
    else:
        raise ValueError(f"Unsupported vq model type: {vq_cfg.type}")
    vq_model.eval()
    vq_model.requires_grad_(False)
    vocab_offset = model.config.vocab_size - model.config.action_vocab_size
    mask_id = model.config.mask_token_id
    mask_schedule = get_mask_schedule(config.training.get("mask_schedule", "cosine"))
    if not hasattr(config.training, "min_masking_rate"):
        config.training.min_masking_rate = 0.0
    dataset_file_paths = config.dataset.file_paths
    if config.training.action_type == "joint":
        robotwin_dataset = [
            EpisodesHDF5Dataset(
                dataset_dir=dataset_file_path,
                chunk_size=config.training.chunk_size,
                vocab_offset=vocab_offset,
                action_vocab_size=action_vocab_size,
                action_type=config.training.action_type,
                action_dim=config.training.action_dim,
                is_clean_data=config.training.is_clean_data,
                select_num=config.training.per_episode,
                input_joint_norm=True,
                joint_action_max=torch.tensor(config.training.joint_action_max),
                joint_action_min=torch.tensor(config.training.joint_action_min),
            )
            for dataset_file_path in dataset_file_paths
        ]

    elif config.training.action_type == "ee":
        robotwin_dataset = [
            EpisodesHDF5Dataset(
                dataset_dir=dataset_file_path,
                chunk_size=config.training.chunk_size,
                vocab_offset=vocab_offset,
                action_vocab_size=action_vocab_size,
                action_type=config.training.action_type,
                action_dim=config.training.action_dim,
                select_num=config.training.per_episode,
                is_clean_data=config.training.is_clean_data,
                predict_image=config.training.co_t2i,
                predict_text=config.training.co_mmu,
                frequency_ratio=config.dataset.frequency_ratio,
                original_dataset_len=config.dataset.original_dataset_len,
            )
            for dataset_file_path in dataset_file_paths
        ]
    else:
        raise ValueError(f"Unsupported action type: {config.training.action_type}")
    robotwin_mix_dataset = ConcatDataset(robotwin_dataset)
    dataloader = DataLoader(
        robotwin_mix_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.dataprocess_nums,
        shuffle=True,
        collate_fn=EpisodesHDF5Dataset.collate_fn,
    )

    #######################################
    #   Random mask for variable length
    ######################################
    @torch.no_grad()
    def prepare_inputs_and_labels_for_mmu_action(batch, eps=1e-3):
        images, tasks, state_tokens, _, _, action_dims, descriptions, _ = batch
        image_tokens = []
        for imgs in images:
            # img_tokens = []
            # for img in imgs:
            #     img = img.to(accelerator.device)
            #     with torch.no_grad():
            #         tokens = vq_model.get_code(img.unsqueeze(0))[0]
            #     tokens = tokens + len(uni_prompting.text_tokenizer)
            #     img_tokens.append(tokens)
            # image_tokens.append(img_tokens)
            imgs = torch.stack(imgs, dim=0).to(accelerator.device)
            with torch.no_grad():
                img_tokens = vq_model.get_code(imgs)
            img_tokens = img_tokens + len(uni_prompting.text_tokenizer)
            # img_tokens = [img_token.tolist() for img_token in img_tokens]
            image_tokens.append(img_tokens)

        input_ids_mmu, prompt_masks, labels_mmu = uni_prompting(
            (
                image_tokens,
                tasks,
                descriptions,
                action_dims,
                accelerator.device,
                state_tokens,
            ),
            "mmu_action",
            config=config,
        )

        # ===== original mmu prepare =====
        b, l = input_ids_mmu.shape
        t = torch.rand(b, device=input_ids_mmu.device)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, l)

        masked_indices = torch.rand((b, l), device=input_ids_mmu.device) < p_mask
        # 126336 is used for [MASK] token
        noisy_batch = torch.where(masked_indices, mask_id, input_ids_mmu)
        masked_indices = noisy_batch == mask_id
        noisy_batch[prompt_masks.bool()] = input_ids_mmu[prompt_masks.bool()]
        masked_indices = noisy_batch == mask_id

        prompt_masks = prompt_masks.to(torch.int64)
        answer_lengths = torch.sum((1 - prompt_masks), dim=-1, keepdim=True)
        answer_lengths = answer_lengths.repeat(1, noisy_batch.shape[1])

        return noisy_batch, labels_mmu, p_mask, answer_lengths

    def prepare_inputs_and_labels(batch, is_train=True, seed=None):
        (
            images,
            texts,
            state_tokens,
            prev_action_tokens,
            action_tokens,
            action_dims,
            _,
            _,
        ) = batch
        image_tokens = []
        for imgs in images:
            # img_tokens = []
            # for img in imgs:
            #     img = img.to(accelerator.device)
            #     with torch.no_grad():
            #         tokens = vq_model.get_code(img.unsqueeze(0))[0]
            #     tokens = tokens + len(uni_prompting.text_tokenizer)
            #     img_tokens.append(tokens)
            # image_tokens.append(img_tokens)
            imgs = torch.stack(imgs, dim=0).to(accelerator.device)
            with torch.no_grad():
                img_tokens = vq_model.get_code(imgs)
            img_tokens = img_tokens + len(uni_prompting.text_tokenizer)
            # img_tokens = [img_token.tolist() for img_token in img_tokens]
            image_tokens.append(img_tokens)

        action_tokens = [a.to(accelerator.device) for a in action_tokens]
        masked_action, labels, _, _ = mask_or_random_replace_action_tokens(
            action_tokens,
            mask_id,
            uni_prompting.pad_id,  # important,use in mm2a_prompt
            config,
            mask_schedule,
            action_vocab_size,
            is_train=is_train,
            seed=seed,
        )
        masked_list, label_list = [], []
        for i, a in enumerate(action_tokens):
            seq_len = a.size(0)
            masked_list.append(masked_action[i, :seq_len].cpu())
            label_list.append(labels[i, :seq_len].cpu())
        input_ids, attention_masks, label_ids = uni_prompting(
            (
                image_tokens,
                texts,
                state_tokens,
                prev_action_tokens,
                masked_list,
                action_dims,
                label_list,
                accelerator.device,
                config.training.chunk_size,
            ),
            "mm2a",
            config=config,
        )
        return (
            input_ids.to(accelerator.device),
            label_ids.to(accelerator.device),
            attention_masks.to(accelerator.device),
        )

    def prepare_inputs_and_labels_image(batch, is_train=True, seed=None):
        images, tasks, state_tokens, _, _, action_dims, _, predict_images = batch
        image_tokens = []
        for imgs in images:
            img_tokens = []
            # for img in imgs:
            #     img = img.to(accelerator.device)
            #     with torch.no_grad():
            #         tokens = vq_model.get_code(img.unsqueeze(0))[0]
            #     tokens = tokens + len(uni_prompting.text_tokenizer)
            #     img_tokens.append(tokens)
            # image_tokens.append(img_tokens)
            imgs = torch.stack(imgs, dim=0).to(accelerator.device)
            with torch.no_grad():
                img_tokens = vq_model.get_code(imgs)
            img_tokens = img_tokens + len(uni_prompting.text_tokenizer)
            # img_tokens = [img_token.tolist() for img_token in img_tokens]
            image_tokens.append(img_tokens)

        predict_image_tokens = []

        predict_images = torch.stack(predict_images, dim=0)
        with torch.no_grad():
            predict_image_tokens = vq_model.get_code(predict_images)
        predict_image_tokens = predict_image_tokens + len(uni_prompting.text_tokenizer)

        masked_list, label_list, loss_weight, mask_prob = mask_or_random_replace_tokens(
            predict_image_tokens,
            mask_id,
            config,
            mask_schedule=mask_schedule,
            is_train=is_train,
            seed=seed,
        )
        input_ids, attention_masks, label_ids = uni_prompting(
            (
                image_tokens,
                tasks,
                masked_list,
                action_dims,
                label_list,
                accelerator.device,
                state_tokens,
            ),
            "t2i_action",
            config=config,
        )
        return input_ids, label_ids, attention_masks, mask_prob

    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer_config = config.optimizer.params
    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    # Create mask scheduler
    if config.get("mask_schedule", None) is not None:
        schedule = config.mask_schedule.schedule
        args = config.mask_schedule.get("params", {})
        mask_schedule = get_mask_schedule(schedule, **args)
    else:
        mask_schedule = get_mask_schedule(
            config.training.get("mask_schedule", "cosine")
        )

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    try:
        max_train_steps = config.training.max_train_steps
    except:
        print("max train steps not set, use total step")
        updates_per_epoch = math.ceil(
            len(dataloader) / accelerator.gradient_accumulation_steps
        )
        max_train_steps = config.training.num_epochs * updates_per_epoch
    try:
        warmup_steps = config.lr_scheduler.params.warmup_steps
    except:
        warmup_steps = int(max_train_steps * 0.08)

    print("total training step:", max_train_steps)
    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=max_train_steps,
        num_warmup_steps=warmup_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale,
    )

    # logger.info("============= Norm =============")
    # logger.info(f"MAX_norm:{robotwin_dataset.joint_action_max}")
    # logger.info(f"MIN_norm:{robotwin_dataset.joint_action_min}")

    model.train()
    log_interval = config.training.get("log_interval", 50)
    logger = setup_logger(output_dir / "train.log", accelerator.is_main_process)
    if accelerator.is_main_process:
        logger.info("======== Training setup ========")
        logger.info("output_dir: %s", output_dir)
        logger.info("num_epochs: %s", config.training.num_epochs)
        logger.info(
            "grad_accum_steps (from accelerate): %s",
            accelerator.gradient_accumulation_steps,
        )
        logger.info("================================")
    ##################################
    #   Training_script   #
    #################################
    global_update_step = 0
    for epoch in range(config.training.num_epochs):

        if accelerator.is_main_process:
            pbar = tqdm(
                total=updates_per_epoch,
                dynamic_ncols=True,
                desc=f"Epoch {epoch}",
                disable=False,
            )
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                input_ids_all = None
                labels_all = None
                device = None

                batch_size_action = 0
                batch_size_t2i = 0
                batch_size_mmu = 0

                attention_masks_action = None
                attention_masks_t2i = None
                p_mask_mmu = None
                answer_lengths_mmu = None

                # --- 1) action ---
                if getattr(config.training, "co_action", False):
                    input_ids_action, labels_action, attention_masks_action = (
                        prepare_inputs_and_labels(batch)
                    )
                    batch_size_action = input_ids_action.shape[0]

                    input_ids_all = input_ids_action
                    labels_all = labels_action
                    device = input_ids_action.device

                # --- 2) t2i ---
                if getattr(config.training, "co_t2i", False):
                    input_ids_t2i, label_ids_t2i, attention_masks_t2i, p_mask_t2i = (
                        prepare_inputs_and_labels_image(batch)
                    )
                    batch_size_t2i = input_ids_t2i.shape[0]

                    if input_ids_all is None:
                        input_ids_all = input_ids_t2i
                        labels_all = label_ids_t2i
                        device = input_ids_t2i.device
                    else:
                        input_ids_all = torch.cat(
                            (input_ids_all, input_ids_t2i.to(device)), dim=0
                        )
                        labels_all = torch.cat(
                            (labels_all, label_ids_t2i.to(device)), dim=0
                        )

                # --- 3) mmu ---
                if getattr(config.training, "co_mmu", False):
                    input_ids_mmu, labels_mmu, p_mask_mmu, answer_lengths_mmu = (
                        prepare_inputs_and_labels_for_mmu_action(
                            batch, eps=config.training.mmu_eps
                        )
                    )
                    batch_size_mmu = input_ids_mmu.shape[0]

                    if input_ids_all is None:
                        input_ids_all = input_ids_mmu
                        labels_all = labels_mmu
                        device = input_ids_mmu.device
                    else:
                        input_ids_all = torch.cat(
                            (input_ids_all, input_ids_mmu.to(device)), dim=0
                        )
                        labels_all = torch.cat(
                            (labels_all, labels_mmu.to(device)), dim=0
                        )

                if input_ids_all is None or labels_all is None:
                    raise ValueError(
                        "No training branches enabled: set at least one of co_action/co_t2i/co_mmu to True."
                    )
                # if need to use other loss, pass corresponding arguments to this function
                logits, loss_action, loss_t2i, loss_lm, loss_mmu = (
                    model.forward_process_vla(
                        input_ids=input_ids_all,
                        labels=labels_all,
                        batch_size_t2i=batch_size_t2i,
                        batch_size_lm=0,
                        batch_size_mmu=batch_size_mmu,
                        batch_size_action=batch_size_action,
                        max_image_seq_length=max_image_seq_length,
                        max_action_prompt_len=max_action_prompt_len,
                        p_mask_mmu=p_mask_mmu,
                        answer_lengths_mmu=answer_lengths_mmu,
                        action_loss_type=config.training.action_loss_type,
                        action_masks=(
                            attention_masks_action if batch_size_action > 0 else None
                        ),
                        t2i_masks=attention_masks_t2i if batch_size_t2i > 0 else None,
                    )
                )

                loss = (
                    config.training.mmu_weight * loss_mmu
                    + config.training.action_weight * loss_action
                    + config.training.t2i_weight * loss_t2i
                )
                accelerator.backward(loss)

            if accelerator.sync_gradients:
                optimizer.step()
                if lr_scheduler is not None and accelerator.sync_gradients:
                    lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update_step += 1
                loss_avg = accelerator.gather_for_metrics(loss.detach()).mean().item()
                loss_mmu_avg = (
                    accelerator.gather_for_metrics(loss_mmu.detach()).mean().item()
                )
                loss_action_avg = (
                    accelerator.gather_for_metrics(loss_action.detach()).mean().item()
                )
                loss_t2i_avg = (
                    accelerator.gather_for_metrics(loss_t2i.detach()).mean().item()
                )
                try:
                    current_lr = (
                        lr_scheduler.get_last_lr()[0]
                        if lr_scheduler is not None
                        else optimizer.param_groups[0]["lr"]
                    )
                except:
                    current_lr = optimizer.param_groups[0]["lr"]

                accelerator.log(
                    {
                        "train/loss": loss_avg,
                        "train/lr": current_lr,
                        "train/loss_mmu": loss_mmu_avg,
                        "train/loss_action": loss_action_avg,
                        "train/loss_t2i": loss_t2i_avg,
                        "train/epoch": epoch,
                        "train/step_in_epoch": step,
                    },
                    step=global_update_step,
                )
                if accelerator.is_main_process:
                    pbar.update(1)
                    pbar.set_postfix_str(
                        f"loss={loss_avg:.4f} | lr={current_lr:.6g} | step={global_update_step}/{updates_per_epoch*config.training.num_epochs}"
                    )

                    if (global_update_step % log_interval) == 0:
                        logger.info(
                            "[epoch %s] step %s/%s | "
                            "epoch_step %s/%s | "
                            "loss %.4f | lr %.6g",
                            epoch,
                            global_update_step,
                            updates_per_epoch * config.training.num_epochs,
                            global_update_step - updates_per_epoch * epoch,
                            updates_per_epoch,
                            loss_avg,
                            current_lr,
                        )
            if getattr(config.training, "save_step", False):
                if global_update_step % config.training.save_step == 0 and step != 0:
                    if accelerator.is_main_process:
                        steps_output_dir = os.path.join(output_dir, f"steps_{step}")
                        accelerator.unwrap_model(model).save_pretrained(
                            steps_output_dir
                        )
                        uni_prompting.text_tokenizer.save_pretrained(steps_output_dir)
        if accelerator.is_main_process:
            pbar.close()
            logger.info("Epoch %s finished.", epoch)
            if epoch + 1 < config.training.num_epochs:
                epoch_output_dir = os.path.join(output_dir, f"checkpoint_epoch_{epoch}")
                accelerator.unwrap_model(model).save_pretrained(epoch_output_dir)
                uni_prompting.text_tokenizer.save_pretrained(epoch_output_dir)

    if accelerator.is_main_process:
        pbar.close()
        accelerator.unwrap_model(model).save_pretrained(output_dir)
        uni_prompting.text_tokenizer.save_pretrained(output_dir)
    accelerator.end_training()


if __name__ == "__main__":
    main()
