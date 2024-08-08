import os
from pathlib import Path

import torch
import tqdm
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from lerobot.common.policies.openvla.language_dataset import (
    LanguageLeRobotDataset,
    PaddedCollatorForActionPrediction,
    PurePromptBuilder,
)
from lerobot.common.policies.openvla.modeling_openvla import ActionTokenizer
from lerobot.common.policies.openvla.utils import compute_action_q01_q99, normalize


class FinetuneConfig:
    vla_path: str = "openvla/openvla-7b"  # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    dataset_repo_id = "m1b/new_setting_red_box"
    wandb: bool = True  # Whether to log to W&B
    job_name: str = "finetune-openvla-koch-pick-lego"  # Name of W&B job
    delta_timestamps = None  # Delta timestamps for action prediction
    run_root_dir: Path = Path(
        "/fsx/marina_barannikov/outputs/openvla/koch_pick_lego/runs"
    )  # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path(
        "/fsx/marina_barannikov/outputs/openvla/koch_pick_lego/adapter-tmp"
    )  # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 16  # Fine-tuning batch size
    max_steps: int = 100000  # Max number of fine-tuning steps
    save_steps: int = 5000  # Interval for checkpoint saving
    learning_rate: float = 2e-5  # Fine-tuning learning rate

    # LoRA Arguments
    use_lora: bool = True  # Whether to use LoRA fine-tuning
    lora_rank: int = 32  # Rank of LoRA weight matrix
    lora_dropout: float = 0.0  # Dropout applied to LoRA weights
    use_quantization: bool = False  # Whether to 4-bit quantize VLA for LoRA fine-tuning
    #   => CAUTION: Reduces memory but hurts performance


def finetune(cfg: FinetuneConfig):
    device = torch.device("cuda")
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    prompt_builder_fn = PurePromptBuilder

    adapter_dir = cfg.adapter_tmp_dir

    run_dir = cfg.run_root_dir
    os.makedirs(run_dir, exist_ok=True)
    print(f"Creating run directory at {run_dir}")

    dataset = LanguageLeRobotDataset(
        cfg.dataset_repo_id,
        delta_timestamps=cfg.delta_timestamps,
        action_tokenizer=action_tokenizer,
        processor=processor,
        prompt_builder_fn=prompt_builder_fn,
    )

    compute_action_q01_q99(dataset)

    dataset = normalize(dataset.stats, dataset)

    quantization_config = None

    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    vla.norm_stats["aloha_sim"] = dataset.stats

    vla = vla.to(device)

    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
    print(f"Created AdamW Optimizer with Learning Rate: {cfg.learning_rate}")

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=4,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )
    if cfg.wandb:
        wandb.init(project="lerobot", name=cfg.job_name, config=cfg)

    step = 0
    vla.train()
    optimizer.zero_grad()
    print(f"Starting Fine-Tuning for {cfg.max_steps} steps...")
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        while step < cfg.max_steps:
            for batch in dataloader:
                if step >= cfg.max_steps:
                    break

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output: CausalLMOutputWithPast = vla(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device),
                        labels=batch["labels"].to(device),
                    )
                    loss = output.loss

                # Backward pass
                loss.backward()

                # Optimizer Step
                optimizer.step()
                optimizer.zero_grad()
                progress.update()

                # Compute Accuracy and L1 Loss for Logging
                action_logits = output.logits[:, vla.vision_backbone.featurizer.patch_embed.num_patches : -1]
                action_preds = action_logits.argmax(dim=2)
                action_gt = batch["labels"][:, 1:].to(action_preds.device)
                mask = action_gt > action_tokenizer.action_token_begin_idx

                # Compute Accuracy
                correct_preds = (action_preds == action_gt) & mask
                action_accuracy = correct_preds.sum().float() / mask.sum().float()

                # Compute L1 Loss on Predicted (Continuous) Actions
                continuous_actions_pred = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                )
                continuous_actions_gt = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                )
                action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

                step += 1

                # Log metrics to W&B
                wandb.log(
                    {
                        "train_loss": loss.item(),
                        "action_accuracy": action_accuracy.item(),
                        "l1_loss": action_l1_loss.item(),
                    },
                    step=step,
                )

                # Save Model Checkpoint
                if step % cfg.save_steps == 0:
                    print(f"Saving Model Checkpoint for Step {step}")
                    step_dir = str(run_dir / f"step-{step}")
                    finetuned_save_dir = str(adapter_dir / f"step-{step}")

                    # Save Processor & Weights
                    processor.save_pretrained(step_dir)
                    vla.save_pretrained(finetuned_save_dir)

                    # Merge LoRA weights into model backbone for faster inference
                    if cfg.use_lora:
                        base_vla = AutoModelForVision2Seq.from_pretrained(
                            cfg.vla_path,
                            torch_dtype=torch.bfloat16,
                            low_cpu_mem_usage=True,
                            trust_remote_code=True,
                        )
                        merged_vla = PeftModel.from_pretrained(base_vla, finetuned_save_dir)
                        merged_vla = merged_vla.merge_and_unload()
                        merged_vla.save_pretrained(step_dir)


if __name__ == "__main__":
    cfg = FinetuneConfig()
    finetune(cfg)
