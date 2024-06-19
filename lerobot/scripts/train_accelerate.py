from pathlib import Path

import hydra
import os

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

# Create a directory to store the training checkpoint.
output_directory = Path("/fsx/marina_barannikov/outputs/train/mixed_precision_test_accelerated_act")
output_directory.mkdir(parents=True, exist_ok=True)

pretrained_model_dir_name = "pretrained_model"
training_state_file_name = "training_state.pth"

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

    
    offline_dataset = make_dataset(cfg)
    accelerator.print(f"Dataset loaded with {len(offline_dataset)} samples")

    eval_env = None
    if cfg.training.eval_freq > 0:
        accelerator.print("make_env")
        eval_env = make_env(cfg)

    policy = make_policy(cfg, dataset_stats=offline_dataset.stats)

    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    num_total_params = sum(p.numel() for p in policy.parameters())
    accelerator.print(f"Policy created with {num_total_params} parameters")

    if cfg.training.get("drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            offline_dataset.episode_data_index,
            drop_n_last_frames=cfg.training.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    train_dataloader = torch.utils.data.DataLoader(
        offline_dataset,
        num_workers=cfg.training.num_workers,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device != torch.device("cpu"),
        drop_last=False,
    )


    policy, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, train_dataloader, lr_scheduler
    )
    policy.to(device)

    step = 0

    if cfg.resume=="true":
        resume_step=int(resume_checkpoint.split("/")[-1])
        accelerator.print(f"Resumed from step: {resume_step}")
        accelerator.load_state(resume_checkpoint)
    else:
        resume_step=0

    done = False
    while not done:
        if step == 0:
            accelerator.print("Start offline training on a fixed dataset")

        policy.train()
        if resume_step>step and len(train_dataloader)%resume_step != 0 :
            # We need to skip steps until we reach the resumed step
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            accelerator.print(f"Skipping {resume_step} steps in the dataloader.")
            step += resume_step
        else:
            # After the first iteration though, we need to go back to the original dataloader
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

            if step % cfg.training.save_freq == 0:
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
                        policy,
                        cfg.eval.n_episodes,
                        videos_dir=out_dir/ "eval" / f"videos_step_{step_identifier}",
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
        resume_checkpoint=None
    )

if __name__ == "__main__":
    train_cli()