import contextlib
import math
import os
import shutil
from pathlib import Path

import pandas as pd
import torch

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import init_hydra_config


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
    val_dataset = make_dataset(cfg, split=f"train[{first_val_frame_index}:]")
    return val_dataset


def evaluate_checkpoint(policy, validation_loader, device):
    policy.eval()
    keys = list(policy.expected_image_keys)
    keys.append("observation.state")
    mse_loss = torch.nn.MSELoss()
    l1_loss = torch.nn.L1Loss()
    mse_losses = []
    l1_losses = []
    for batch in validation_loader:
        if isinstance(policy, DiffusionPolicy):
            # For diffusion models, we need a single frame of the observation and action
            batch["observation.state"] = torch.squeeze(batch["observation.state"], dim=1)
            batch["observation.image"] = torch.squeeze(batch["observation.image"], dim=1)
            batch["action"] = torch.squeeze(batch["action"], dim=1)
        else:
            batch["action"] = torch.squeeze(batch["action"])
        observation = {key: batch[key].to(device, non_blocking=True) for key in keys}
        batch["action"] = batch["action"].to(device, non_blocking=True)
        batch = policy.normalize_targets(batch)
        actions = batch["action"]
        with torch.no_grad():
            predicted_actions = policy.select_action(observation)
            mse_losses.append(mse_loss(predicted_actions, actions).item())
            l1_losses.append(l1_loss(predicted_actions, actions).item())

    return sum(mse_losses) / len(mse_losses), sum(l1_losses) / len(l1_losses)


def main(hydra_cfg, list_of_dirs):
    model_json = "/fsx/marina_barannikov/outputs/train/compare_val_loss/pusht_84/020000/config.json"
    device = hydra_cfg.device

    val_dataset = get_train_val_datasets(hydra_cfg, split_value=0.8)
    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    results = []
    for root_dir in list_of_dirs:
        for subdir in os.listdir(root_dir):
            if subdir.isdigit():  # Check if the subdirectory name is numerical
                subdir_path = os.path.join(root_dir, subdir)
                checkpoint_path = os.path.join(subdir_path, "model.safetensors")
                if os.path.isfile(checkpoint_path):
                    with contextlib.suppress(shutil.SameFileError):
                        shutil.copy(model_json, subdir_path)
                    print(f"Evaluating checkpoint from {checkpoint_path}")
                    policy = make_policy(hydra_cfg=hydra_cfg, pretrained_policy_name_or_path=subdir_path)
                    mse, l1 = evaluate_checkpoint(policy, validation_loader, device)
                    results.append(
                        {"root_checkpoint": root_dir, "l1_loss": l1, "step": int(subdir), "mse_loss": mse}
                    )

    df = pd.DataFrame(results)

    # Save DataFrame to a CSV file
    output_file = "dev/pusht_normalized_results.csv"
    df.to_csv(output_file, index=False)

    print(f"Results saved to {output_file}")

    return


if __name__ == "__main__":
    list_of_dirs = [
        "/fsx/marina_barannikov/outputs/train/compare_val_loss/pusht_85",
        "/fsx/marina_barannikov/outputs/train/compare_val_loss/pusht_84",
        "/fsx/marina_barannikov/outputs/train/compare_val_loss/pusht_100000",
    ]
    policy_path = Path("/fsx/marina_barannikov/outputs/train/compare_val_loss/pusht_84/010000")
    cfg = init_hydra_config(str(policy_path / "config.yaml"))
    if cfg.policy.name == "diffusion":
        cfg.training.delta_timestamps["observation.state"] = [0]
        cfg.training.delta_timestamps["observation.image"] = [0]
        cfg.training.delta_timestamps["action"] = [i / 10 for i in range(1)]
    main(cfg, list_of_dirs)
