import os
from pathlib import Path
from typing import Callable

import torch
import torch.utils
from PIL import Image
from torchvision.transforms import ToPILImage

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

DATA_DIR = Path(os.environ["DATA_DIR"]) if "DATA_DIR" in os.environ else None
CODEBASE_VERSION = "v1.4"

instruction = "put the cube into the red box"
main_camera_key = "phone"
to_pil = ToPILImage()


def tensor_to_image(tensor: torch.Tensor) -> Image:
    """Converts a torch tensor to a PIL image."""
    return to_pil(tensor)


class LanguageLeRobotDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        root: Path | None = None,
        split: str = "train",
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        prompt_builder_fn: Callable = None,
        action_tokenizer: Callable = None,
        processor: Callable = None,
    ):
        super().__init__(repo_id, root, split, image_transforms, delta_timestamps)
        self.prompt_builder_fn = prompt_builder_fn
        self.action_tokenizer = action_tokenizer
        self.processor = processor
        self.IGNORE_INDEX = -100

    def __getitem__(self, idx):
        item = super().__getitem__(idx)

        action = item["action"]
        img = tensor_to_image(item[f"observation.images.{main_camera_key}"])
        lang = instruction

        # Construct Chat-based Prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize
        input_ids = self.processor.tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.processor.image_processor(img, return_tensors="pt").pixel_values[0]

        # Ignore unnecessary parts in the labels
        labels[: -(len(action) + 1)] = self.IGNORE_INDEX
        labels[-1] = self.IGNORE_INDEX

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
            "dataset_name": self.repo_id,
            "action": action,
            "next.done": item["next.done"],
        }
