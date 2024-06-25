

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset
import torch

d1="lerobot/aloha_sim_insertion_human"
d2="lerobot/aloha_sim_transfer_cube_human"
dataset1 = LeRobotDataset(d1)
dataset2 = LeRobotDataset(d2)

multidataset = MultiLeRobotDataset([d1, d2])

print(vars(multidataset))
print(multidataset._datasets[0].num_samples)
print(multidataset.repo_id_to_index)