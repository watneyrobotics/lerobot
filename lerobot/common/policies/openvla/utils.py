import torch


def compute_action_q01_q99(dataset):
    print("Computing action q01 and q99 quantiles")
    t = dataset.hf_dataset["action"]
    t = torch.stack(t, 0)
    q = torch.quantile(t, torch.tensor([0.01, 0.99]), dim=0)

    dataset.stats["action"]["q01"] = q[0]
    dataset.stats["action"]["q99"] = q[1]

    return dataset


def normalize(metadata, dataset):
    keys_to_normalize = {
        "action": "action",
    }

    # Convert metadata to tensors
    low = {}
    high = {}
    mask = {}
    zeros_mask = {}

    for key, traj_key in keys_to_normalize.items():
        low[traj_key] = torch.tensor(metadata[key]["q01"])
        high[traj_key] = torch.tensor(metadata[key]["q99"])
        mask[traj_key] = torch.tensor(
            metadata[key].get("mask", torch.ones_like(high[traj_key], dtype=torch.bool)), dtype=torch.bool
        )
        zeros_mask[traj_key] = torch.tensor(metadata[key]["min"] == metadata[key]["max"], dtype=torch.bool)

    def normalize_sample(sample):
        for _, traj_key in keys_to_normalize.items():
            sample[traj_key] = torch.where(
                mask[traj_key],
                torch.clamp(
                    2 * (sample[traj_key] - low[traj_key]) / (high[traj_key] - low[traj_key] + 1e-8) - 1,
                    -1,
                    1,
                ),
                sample[traj_key],
            )
            sample[traj_key] = torch.where(
                zeros_mask[traj_key], torch.tensor(0.0, dtype=sample[traj_key].dtype), sample[traj_key]
            )
        return sample

    dataset.hf_dataset = dataset.hf_dataset.map(normalize_sample, num_proc=1, writer_batch_size=32)

    return dataset
