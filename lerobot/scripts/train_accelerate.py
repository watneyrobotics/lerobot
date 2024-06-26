from pathlib import Path

import hydra
import os
import math

import torch
import torch.utils

from lerobot.common.utils.utils import set_global_seed, format_big_number
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.envs.factory import make_env

from lerobot.scripts.train import make_optimizer_and_scheduler
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.scripts.eval import eval_policy

from accelerate import Accelerator

from omegaconf import OmegaConf, DictConfig


pretrained_model_dir_name = "pretrained_model"
training_state_file_name = "training_state.pth"


def get_train_val_datasets(cfg, split_value=0.9):
    # Load the last 10% of episodes of the dataset as a validation set.
    # - Load full dataset
    full_dataset = make_dataset(cfg, split="train")
    # - Calculate train and val subsets
    num_train_episodes = math.floor(full_dataset.num_episodes * split_value)
    num_val_episodes = full_dataset.num_episodes - num_train_episodes
    print(f"Number of episodes in full dataset: {full_dataset.num_episodes}")
    print(f"Number of episodes in training dataset ({split_value*100}% subset): {num_train_episodes}")
    print(f"Number of episodes in validation dataset ({split_value*100}% subset): {num_val_episodes}")
    # - Get first frame index of the validation set
    first_val_frame_index = full_dataset.episode_data_index["from"][num_train_episodes].item()
    train_dataset = make_dataset(
        cfg, split=f"train[:{first_val_frame_index}]"
    )
    val_dataset = make_dataset(
        cfg, split=f"train[{first_val_frame_index}:]"
    )
    return train_dataset, val_dataset




def train(cfg: DictConfig, job_name, out_dir, resume_checkpoint=None):
    if out_dir is None:
        raise NotImplementedError()
    if job_name is None:
        raise NotImplementedError()
    
    out_dir=Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(log_with="wandb")

    accelerator.init_trackers(
        project_name="lerobot",
        init_kwargs={"wandb": {"name":job_name, "job_type": "train", "config": OmegaConf.to_container(cfg, resolve=True)}}
    )
    # Check device is available
    device = accelerator.device
    print(device)

    set_global_seed(cfg.seed)
    accelerator.print(f"Global seed set to {cfg.seed}")

    train_dataset, val_dataset = get_train_val_datasets(cfg, split_value=0.9)

    accelerator.print(f"Number of frames in training dataset (90% subset): {len(train_dataset)}")
    accelerator.print(f"Number of frames in validation dataset (10% subset): {len(val_dataset)}")

    eval_env = None
    if cfg.training.eval_freq > 0:
        accelerator.print("make_env")
        eval_env = make_env(cfg)

    policy = make_policy(cfg, dataset_stats=train_dataset.stats)

    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    num_total_params = sum(p.numel() for p in policy.parameters())
    accelerator.print(f"Policy created with {num_total_params} parameters")

    if cfg.training.get("drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            train_dataset.episode_data_index,
            drop_n_last_frames=cfg.training.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=cfg.training.num_workers,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device != torch.device("cpu"),
        drop_last=False,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        num_workers=cfg.training.num_workers,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        pin_memory=device != torch.device("cpu"),
        drop_last=False,
    )


    policy, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    policy.to(device)

    step = 0

    if cfg.resume==True or cfg.resume=="true":
        subfolders = [p for p in out_dir.iterdir() if p.is_dir()]
        latest_subfolder = max(subfolders, key=lambda p: os.path.getctime(str(p)))
        accelerator.print(f"Resuming from {latest_subfolder}")
        resume_step=int(latest_subfolder.name) + 1
        accelerator.print(f"Resumed from step: {resume_step}")
        accelerator.load_state(latest_subfolder)
    else:
        accelerator.print("Starting from scratch")
        resume_step=0

    step = resume_step

    done = False
    while not done:
        if step == 0:
            accelerator.print("Start offline training on a fixed dataset")

        policy.train()

        active_dataloader = train_dataloader

        for batch in active_dataloader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            output_dict = policy.forward(batch)
            loss = output_dict["loss"]
            accelerator.backward(loss)

            grad_norm = accelerator.clip_grad_norm_(policy.parameters(), cfg.training.grad_clip_norm)
            
            optimizer.step()
            optimizer.zero_grad()

            if lr_scheduler is not None:
                lr_scheduler.step()

            if step % cfg.training.log_freq == 0:
                accelerator.print(f"step: {step} loss: {loss.item():.3f}")
                accelerator.log({"train/loss": loss.item()}, step=step)
                accelerator.log({"train/grad_norm": grad_norm}, step=step)
                for k, v in output_dict.items():
                    if k != "loss":
                        accelerator.log({f"train/{k}": v}, step=step)

            if step % cfg.training.save_freq == 0 and step > 0:
                _num_digits = max(6, len(str(cfg.training.offline_steps + cfg.training.online_steps)))
                step_identifier = f"{step:0{_num_digits}d}"
                save_dir = out_dir / step_identifier
                accelerator.print(f"Saving state to {save_dir}")
                accelerator.save_state(save_dir)
                OmegaConf.save(cfg, save_dir / "config.yaml")
            
            if cfg.training.eval_freq > 0 and step % cfg.training.eval_freq == 0:
                accelerator.print(f"Evaluating policy from process {accelerator.local_process_index}")
                with torch.no_grad():
                    accelerator.wait_for_everyone()
                    eval_info = eval_policy(
                        eval_env,
                        accelerator.unwrap_model(policy),
                        cfg.eval.n_episodes,
                        videos_dir=out_dir/ "eval" / f"videos_step_{step}",
                        max_episodes_rendered=4,
                        enable_progbar=True,
                        start_seed=cfg.seed,
                    )

                    for k, v in eval_info.items():
                        accelerator.print({f"eval/{k}": v}, step=step+1)
                        if not isinstance(v, (int, float)):
                            accelerator.print(f"Skipping {k} from logging because it is not a scalar")
                            continue
                        accelerator.log({f"eval/{k}": v}, step=step+1)
                    accelerator.print(f"Evaluated policy from process {accelerator.local_process_index}")

            step += 1

            if step > cfg.training.offline_steps:
                done = True
                break
                            
            
        policy.eval()

        total_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                output_dict = policy.forward(batch)
                loss = output_dict["loss"]
                total_loss += loss.item()

        average_loss = total_loss / len(val_dataloader)
        accelerator.print(f"Validation loss: {average_loss:.3f}")
        accelerator.log({"val/loss": average_loss}, step=step)
            

    unwrapped_policy = accelerator.unwrap_model(policy)
    unwrapped_policy.save_pretrained(out_dir / "final")
    accelerator.print("Finished offline training")
    accelerator.end_training()
    
@hydra.main(version_base="1.2", config_name="default", config_path="../configs")
def train_cli(cfg: dict):
    train(
        cfg,
        job_name=hydra.core.hydra_config.HydraConfig.get().job.name,
        out_dir=hydra.core.hydra_config.HydraConfig.get().run.dir,
    )

if __name__ == "__main__":
    train_cli()

