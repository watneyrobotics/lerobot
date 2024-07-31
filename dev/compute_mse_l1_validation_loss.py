import json
import math
import os
from pathlib import Path

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
            batch["observation.state"] = torch.squeeze(batch["observation.state"])
            batch["observation.image"] = torch.squeeze(batch["observation.image"])
        actions = batch["action"]
        observation = {key: batch[key].to(device, non_blocking=True) for key in keys}
        actions = actions.to(device)
        with torch.no_grad():
            policy.reset()
            predicted_actions = policy.select_action(observation)
            print("Calculating MSE and L1 losses")
            mse_losses.append(mse_loss(predicted_actions, actions).item())
            l1_losses.append(l1_loss(predicted_actions, actions).item())

    return sum(mse_losses) / len(mse_losses), sum(l1_losses) / len(l1_losses)


def main(hydra_cfg, root_dir, output_file):
    device = hydra_cfg.device

    val_dataset = get_train_val_datasets(hydra_cfg, split_value=0.8)
    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)

    results = []
    for subdir in os.listdir(root_dir):
        if subdir.isdigit():  # Check if the subdirectory name is numerical
            subdir_path = os.path.join(root_dir, subdir)
            checkpoint_path = os.path.join(subdir_path, "model.safetensors")
            if os.path.isfile(checkpoint_path):
                print(f"Loading checkpoint from {checkpoint_path}")
                policy = make_policy(hydra_cfg=hydra_cfg, pretrained_policy_name_or_path=subdir_path)
                print("Evaluating checkpoint")
                mse, l1 = evaluate_checkpoint(policy, validation_loader, device)
                results.append(
                    {"checkpoint": subdir_path, "l1_loss": l1, "step": int(subdir), "mse_loss": mse}
                )

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    return


if __name__ == "__main__":
    policy_path = Path("/fsx/marina_barannikov/outputs/train/compare_val_loss/pusht_84/010000")
    cfg = init_hydra_config(str(policy_path / "config.yaml"))
    cfg.training.delta_timestamps["observation.image"] = [0]
    cfg.training.delta_timestamps["action"] = [i / 10 for i in range(8)]
    cfg.training.delta_timestamps["observation.state"] = [0]
    main(cfg, "/fsx/marina_barannikov/outputs/train/compare_val_loss/pusht_84", "dev/test.json")
