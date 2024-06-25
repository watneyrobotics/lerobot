from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import torch


repo_id = "lerobot/aloha_static_tape"
dataset=LeRobotDataset(repo_id)

item=dataset[0]

dict_obs={key: item[key] for key in item.keys() if "observation.image" in key or "state" in key}

def main(observation, stats):
    """
    Args: observation: A dictionary containing the observation data, includes the state & image
    """
    threshold = 2

    for key in observation:

        z_score = torch.abs(observation[key] - stats[key]['mean']) / stats[key]['std']
        outlier_mask = z_score > threshold

        num_outlier_dims = (torch.abs(z_score) > threshold).sum().squeeze()
        total_elements = outlier_mask.numel()
        total_z_score = outlier_mask.sum().item()

        if total_z_score / total_elements > 0.5:
            print(f"Warning: More than 50% of the elements in {key} are outliers") 
            print(f"Number of outlier dimensions: {num_outlier_dims}")
        else:
            print(f"No outliers detected in {key}")

main(dict_obs, dataset.stats)