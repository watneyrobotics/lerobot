import json
import os
from copy import deepcopy
from math import ceil

import einops
import numpy as np
import torch
import tqdm


def get_dataset_statistics(dataset, save_dir, batch_size=32, num_workers=16, max_num_samples=None):
    """Compute mean/std and min/max statistics of action key in a dataset, and zeroed stats for proprio."""
    if os.path.exists(save_dir):
        print(f"Loading dataset statistics from {save_dir}")
        with open(save_dir) as f:
            return json.load(f)

    if max_num_samples is None:
        max_num_samples = len(dataset)

    # Only computing stats for 'action'
    stats_patterns = {"action": "b c -> c"}

    # Initialize mean, std, max, and min for 'action'
    mean, std, max, min = {}, {}, {}, {}
    for key in stats_patterns:
        mean[key] = torch.tensor(0.0).float()
        std[key] = torch.tensor(0.0).float()
        max[key] = torch.tensor(-float("inf")).float()
        min[key] = torch.tensor(float("inf")).float()

    def create_seeded_dataloader(dataset, batch_size, seed):
        generator = torch.Generator()
        generator.manual_seed(seed)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=generator,
        )
        return dataloader

    first_batch = None
    dataloader = create_seeded_dataloader(dataset, batch_size, seed=84)
    num_transitions, num_trajectories = 0, 0

    for i, batch in enumerate(
        tqdm.tqdm(
            dataloader, total=ceil(max_num_samples / batch_size), desc="Compute mean, min, max, q01, q99"
        )
    ):
        this_batch_size = len(batch["action"])
        running_item_count += this_batch_size
        if first_batch is None:
            first_batch = deepcopy(batch)

        for key, pattern in stats_patterns.items():
            batch[key] = batch[key].float()
            # Numerically stable update step for mean computation.
            batch_mean = einops.reduce(batch[key], pattern, "mean")
            mean[key] = mean[key] + this_batch_size * (batch_mean - mean[key]) / running_item_count
            max[key] = torch.maximum(max[key], einops.reduce(batch[key], pattern, "max"))
            min[key] = torch.minimum(min[key], einops.reduce(batch[key], pattern, "min"))

        num_transitions += this_batch_size
        if "next.done" in batch and batch["next.done"].sum() > 0:
            num_trajectories += batch["next.done"].sum().item()

        if i == ceil(max_num_samples / batch_size) - 1:
            break

    # Create zeroed stats for 'proprio' based on the shape of 'action' stats
    action_shape = mean["action"].shape
    zero_stats = {
        "mean": torch.zeros(action_shape).tolist(),
        "max": torch.zeros(action_shape).tolist(),
        "min": torch.zeros(action_shape).tolist(),
        "q01": torch.zeros(action_shape).tolist(),
        "q99": torch.zeros(action_shape).tolist(),
    }

    # Converting tensors to lists for JSON serialization
    stats = {
        "action": {
            "mean": mean["action"].tolist(),
            "max": max["action"].tolist(),
            "min": min["action"].tolist(),
            "q01": np.quantile(mean["action"], 0.01).tolist(),
            "q99": np.quantile(mean["action"], 0.99).tolist(),
        },
        "proprio": zero_stats,
        "num_transitions": num_transitions,
        "num_trajectories": num_trajectories,
    }

    # Save the computed statistics to a file
    with open(save_dir, "w") as f:
        json.dump(stats, f)

    return stats


def compute_q01_q99(dataset):
    t = dataset.hf_dataset["action"]
    
    q01_per_dimension = np.zeros(t.shape[1])
    q99_per_dimension = np.zeros(t.shape[1])

    for col_idx in range(t.shape[1]):
        current_col = t[:, col_idx].copy()
        current_col.sort()

        q01_per_dimension[col_idx] = np.quantile(t[:, col_idx], 0.01)
        q99_per_dimension[col_idx] = np.quantile(t[:, col_idx], 0.99)

    dataset.stats["action"]["q01"] = q01_per_dimension
    dataset.stats["action"]["q99"] = q99_per_dimension

    return dataset