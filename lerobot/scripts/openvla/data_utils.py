import json
import torch
import os
import numpy as np
from tqdm import tqdm


def get_dataset_statistics(dataset, save_dir):
    path = os.path.join(save_dir, f"dataset_statistics.json")
    print(f"Got stats : {dataset.stats}")
    cardinality = len(dataset)
    print(f"Computing dataset statistics for {cardinality} trajectories.")
    
    print("Computing dataset statistics. This may take a bit, but should only need to happen once.")
    actions, proprios, num_transitions, num_trajectories = [], [], 0, 0
    action_zeros = torch.zeros_like(dataset.hf_dataset[0]["action"])
    for idx in tqdm(range(cardinality)):
        traj = dataset[idx]
        actions.append(traj["action"])
        proprios.append(action_zeros)
        num_transitions += 1
        if traj["next.done"]:
            num_trajectories += 1

    actions, proprios = torch.stack(actions, dim=0), torch.stack(proprios, dim=0)
    print(actions.shape, proprios.shape)

    actions_max, _ = actions.max(dim=0)
    actions_min, _ = actions.min(dim=0)


    metadata = {
        "action": {
            "mean": actions.mean(0).tolist(),
            "std": actions.std().tolist(),
            "max": actions_max.tolist(),
            "min": actions_min.tolist(),
            "q01": np.quantile(actions, 0.01, axis=0).tolist(),
            "q99": np.quantile(actions, 0.99, axis=0).tolist(),
        },
        "proprio": {
            "mean": action_zeros.tolist(),
            "std": action_zeros.tolist(),
            "min": action_zeros.tolist(),
            "q01": np.quantile(proprios, 0.01, axis=0).tolist(),
            "q99": np.quantile(proprios, 0.99, axis=0).tolist(),
        },
        "num_transitions": num_transitions,
        "num_trajectories": num_trajectories,
    }

    with open(save_dir, "w") as f:
        json.dump(metadata, f)

    return metadata