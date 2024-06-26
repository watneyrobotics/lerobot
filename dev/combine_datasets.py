import os
import shutil

def consolidate_episodes(root_folder, episode_count):
    pickup_folder = os.path.join(root_folder, "pickup")
    os.makedirs(pickup_folder, exist_ok=True)

    os.makedirs(os.path.join(pickup_folder, "videos"), exist_ok=True)
    os.makedirs(os.path.join(pickup_folder, "metadata"), exist_ok=True)
    os.makedirs(os.path.join(pickup_folder, "train"), exist_ok=True)

    metadata_folder = os.path.join(os.path.join(root_folder, f"episode_000"), "meta_data")
    metadata_json = os.path.join(metadata_folder, "info.json")
    shutil.copy(metadata_json, os.path.join(pickup_folder, "meta_data"))

    
    for i in range(episode_count):
        episode_folder = os.path.join(root_folder, f"episode_{i:03d}")
        train_folder = os.path.join(episode_folder, "train")
        videos_folder = os.path.join(episode_folder, "videos")

        # Copy train/state.json or *.arrow to pickup/train/
        train_files = os.listdir(train_folder)
        for file in train_files:
            if file.endswith(".arrow"):
                new_filename = f"episode_{i:03d}.arrow"
                shutil.copy(os.path.join(train_folder, file), os.path.join(pickup_folder, "train", new_filename))
                break

        # Copy videos/*.mp4 to pickup/videos/
        video_files = os.listdir(videos_folder)
        for file in video_files:
            if file.endswith(".mp4"):
                shutil.copy(os.path.join(videos_folder, file), os.path.join(pickup_folder, "videos"))

    print("Episodes consolidated successfully into 'pickup' folder.")

# Example usage:
if __name__ == "__main__":
    root_folder = "/Users/mbar/Desktop/projects/huggingface/lerobot/tmp/data/koch/pick_trash1"  # Replace with your root folder containing episode{i} folders
    episode_count = 46  # Replace with the actual number of episodes you have
    consolidate_episodes(root_folder, episode_count)