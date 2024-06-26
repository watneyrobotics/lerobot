import os
import torch
from safetensors import safe_open
from safetensors.torch import save_file

def load_episode_data_index(file_path):
    # Load the safetensors file
    with safe_open(file_path, framework="torch") as f:
        return {key: f[key] for key in f.keys()}


def load_episode_data_index(file_path):
    # Load the safetensors file correctly
    with safe_open(file_path, framework="pt") as f:
        episode_data_index = {key: f.get_tensor(key) for key in f.keys()}
    return episode_data_index

def consolidate_episode_data_indices(root_folder, episode_count):
    combined_episode_data_index = {"from": [], "to": []}
    current_start_index = 0

    for i in range(episode_count):
        episode_folder = os.path.join(root_folder, f"episode_{i:03d}")
        safetensors_file = os.path.join(episode_folder, "meta_data", "episode_data_index.safetensors")

        if os.path.exists(safetensors_file):
            episode_data_index = load_episode_data_index(safetensors_file)

            if episode_data_index:
                episode_length = episode_data_index["to"][0].item() - episode_data_index["from"][0].item() + 1
                combined_episode_data_index["from"].append(current_start_index)
                current_end_index = current_start_index + episode_length - 1
                combined_episode_data_index["to"].append(current_end_index)
                current_start_index = current_end_index + 1
    
    print(f"Indices of all episodes : {combined_episode_data_index}")

    # Convert lists to tensors
    combined_episode_data_index["from"] = torch.tensor(combined_episode_data_index["from"])
    combined_episode_data_index["to"] = torch.tensor(combined_episode_data_index["to"])

    # Save the combined episode data index
    combined_safetensors_file = os.path.join(root_folder, "pickup", "meta_data", "episode_data_index.safetensors")
    save_file(combined_episode_data_index, combined_safetensors_file)

    print(f"Combined episode data index saved to {combined_safetensors_file}")

# Call the function after consolidating episodes
root_folder = "/Users/mbar/Desktop/projects/huggingface/lerobot/tmp/data/koch/pick_trash1"
episode_count = 46  # Number of episodes to consolidate
consolidate_episode_data_indices(root_folder, episode_count)