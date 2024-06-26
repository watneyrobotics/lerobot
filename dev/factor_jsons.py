import os
from datasets import concatenate_datasets, Dataset, Features, Value, Sequence
from datasets import load_dataset
from lerobot.common.datasets.video_utils import VideoFrame

# Path to the train folder containing the .arrow files
train_folder = "/Users/mbar/Desktop/projects/huggingface/lerobot/tmp/data/koch/pick_trash1/pickup/train/"
features = {"observation.images.cam": VideoFrame()}
features = Features({
    "observation.images.cam": VideoFrame(),
    "observation.state": Sequence(feature=Value(dtype="float32", id=None)),
    "action": Sequence(feature=Value(dtype="float32", id=None)),
    "episode_index": Value(dtype="int64", id=None),
    "frame_index": Value(dtype="int64", id=None),
    "timestamp": Value(dtype="float32", id=None),
    "next.done": Value(dtype="bool", id=None),
    "index": Value(dtype="int64", id=None),
})
# List all .arrow files in the train folder
arrow_files = [f for f in os.listdir(train_folder) if f.endswith(".arrow")]

dataset = load_dataset("arrow", data_files={'train': os.path.join(train_folder, arrow_files[0])}, split="train", features=Features(features))

ds = concatenate_datasets([Dataset.from_file(os.path.join(train_folder, arrow_file), split="train") for arrow_file in arrow_files])
ds.save_to_disk("/Users/mbar/Desktop/projects/huggingface/lerobot/tmp/data/koch/pick_trash1/pickup/train/concatenated")
