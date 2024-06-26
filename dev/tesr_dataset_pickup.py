from pathlib import Path
from pprint import pprint

import imageio
import torch

import lerobot
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Let's take one for this example
repo_id = "pickup"

# You can easily load a dataset from a Hugging Face repository
dataset = LeRobotDataset(repo_id, root="tmp/data/koch/pick_trash1")

# LeRobotDataset is actually a thin wrapper around an underlying Hugging Face dataset
# (see https://huggingface.co/docs/datasets/index for more information).
print(dataset)
print(dataset.hf_dataset)

# And provides additional utilities for robotics and compatibility with Pytorch
print(f"\naverage number of frames per episode: {dataset.num_samples / dataset.num_episodes:.3f}")
print(f"frames per second used during data collection: {dataset.fps=}")
print(f"keys to access images from cameras: {dataset.camera_keys=}\n")