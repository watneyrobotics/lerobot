import torch


def compute_action_q01_q99(dataset):
    print("Computing action q01 and q99 quantiles")
    t = dataset.hf_dataset["action"]
    t = torch.stack(t, 0)
    q = torch.quantile(t, torch.tensor([0.01, 0.99]), dim=0)

    dataset.stats["action"]["q01"] = q[0]
    dataset.stats["action"]["q99"] = q[1]

    return dataset
