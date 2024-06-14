#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Visualize data of **all** frames of any episode of a dataset of type LeRobotDataset.

Note: The last frame of the episode doesnt always correspond to a final state.
That's because our datasets are composed of transition from state to state up to
the antepenultimate state associated to the ultimate action to arrive in the final state.
However, there might not be a transition from a final state to another state.

Note: This script aims to visualize the data used to train the neural networks.
~What you see is what you get~. When visualizing image modality, it is often expected to observe
lossly compression artifacts since these images have been decoded from compressed mp4 videos to
save disk space. The compression factor applied has been tuned to not affect success rate.

Examples:

- Visualize data stored on a local machine:
```
local$ python lerobot/scripts/visualize_dataset.py \
    --repo-id lerobot/pusht

local$ open http://localhost:9090
```

- Visualize data stored on a distant machine with a local viewer:
```
distant$ python lerobot/scripts/visualize_dataset.py \
    --repo-id lerobot/pusht

local$ ssh -L 9090:localhost:9090 distant  # create a ssh tunnel
local$ open http://localhost:9090
```

- Select episodes to visualize:
```
python lerobot/scripts/visualize_dataset.py \
    --repo-id lerobot/pusht \
    --episode-indices 7 3 5 1 4
```
"""

import argparse
import http.server
import logging
import os
import shutil
import socketserver
from pathlib import Path
from typing import Iterator

import numpy as np
import rerun as rr
import torch
import torch.utils.data
import tqdm
import yaml
from bs4 import BeautifulSoup
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.utils.utils import init_logging


class EpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self) -> Iterator:
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    assert chw_float32_torch.dtype == torch.float32
    assert chw_float32_torch.ndim == 3
    c, h, w = chw_float32_torch.shape
    assert c < h and c < w, f"expect channel first images, but instead {chw_float32_torch.shape}"
    hwc_uint8_numpy = (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
    return hwc_uint8_numpy


def visualize_dataset(
    repo_id: str,
    episode_index: int,
    batch_size: int = 32,
    num_workers: int = 0,
    mode: str = "local",
    web_port: int = 9090,
    ws_port: int = 9087,
    save: bool = False,
    output_dir: Path | None = None,
    root: Path | None = None,
) -> Path | None:
    if save:
        assert (
            output_dir is not None
        ), "Set an output directory where to write .rrd files with `--output-dir path/to/directory`."

    logging.info("Loading dataset")
    dataset = LeRobotDataset(repo_id, root=root)

    logging.info("Loading dataloader")
    episode_sampler = EpisodeSampler(dataset, episode_index)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        sampler=episode_sampler,
    )

    logging.info("Running inference")
    inference_results = {}
    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.inference_mode():
            output_dict = policy.forward(batch)

        for key in output_dict:
            if key not in inference_results:
                inference_results[key] = []
            inference_results[key].append(output_dict[key].to("cpu"))

    for key in inference_results:
        inference_results[key] = torch.cat(inference_results[key])

    return inference_results


def visualize_dataset(
    repo_id: str,
    episode_indices: list[int] = None,
    output_dir: Path | None = None,
    serve: bool = True,
    port: int = 9090,
    force_overwrite: bool = True,
    policy_repo_id: str | None = None,
    policy_ckpt_path: Path | None = None,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Path | None:
    init_logging()

    has_policy = policy_repo_id or policy_ckpt_path

    if has_policy:
        logging.info("Loading policy")
        if policy_repo_id:
            pretrained_policy_path = Path(snapshot_download(policy_repo_id))
        elif policy_ckpt_path:
            pretrained_policy_path = Path(policy_ckpt_path)
        policy = ACTPolicy.from_pretrained(pretrained_policy_path)
        with open(pretrained_policy_path / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        delta_timestamps = cfg["training"]["delta_timestamps"]
    else:
        delta_timestamps = None

    logging.info("Loading dataset")
    dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)

    if not dataset.video:
        raise NotImplementedError(f"Image datasets ({dataset.video=}) are currently not supported.")

    if output_dir is None:
        output_dir = f"outputs/visualize_dataset/{repo_id}"

    output_dir = Path(output_dir)
    if force_overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a simlink from the dataset video folder containg mp4 files to the output directory
    # so that the http server can get access to the mp4 files.
    ln_videos_dir = output_dir / "videos"
    if not ln_videos_dir.exists():
        ln_videos_dir.symlink_to(dataset.videos_dir.resolve())

    if episode_indices is None:
        episode_indices = list(range(dataset.num_episodes))

    logging.info("Writing html")
    ep_html_fnames = []
    for episode_index in tqdm.tqdm(episode_indices):
        inference_results = None
        if has_policy:
            inference_results_path = output_dir / f"episode_{episode_index}.safetensors"
            if inference_results_path.exists():
                inference_results = load_file(inference_results_path)
            else:
                inference_results = run_inference(dataset, episode_index, policy)
                save_file(inference_results, inference_results_path)

        # write states and actions in a csv
        ep_csv_fname = f"episode_{episode_index}.csv"
        write_episode_data_csv(output_dir, ep_csv_fname, episode_index, dataset, inference_results)

        js_fname = f"episode_{episode_index}.js"
        write_episode_data_js(output_dir, js_fname, ep_csv_fname, dataset)

        # write a html page to view videos and timeseries
        ep_html_fname = f"episode_{episode_index}.html"
        write_episode_data_html(output_dir, ep_html_fname, js_fname, episode_index, dataset)
        ep_html_fnames.append(ep_html_fname)

    write_episodes_list_html(output_dir, "index.html", episode_indices, ep_html_fnames, dataset)

    if serve:
        run_server(output_dir, port)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Name of hugging face repositery containing a LeRobotDataset dataset (e.g. `lerobot/pusht` for https://huggingface.co/datasets/lerobot/pusht).",
    )
    parser.add_argument(
        "--episode-indices",
        type=int,
        nargs="*",
        default=None,
        help="Episode indices to visualize (e.g. `0 1 5 6` to load episodes of index 0, 1, 5 and 6). By default loads all episodes.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory path to write html files and kickoff a web server. By default write them to 'outputs/visualize_dataset/REPO_ID'.",
    )
    parser.add_argument(
        "--serve",
        type=int,
        default=1,
        help="Launch web server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9090,
        help="Web port used by the http server.",
    )
    parser.add_argument(
        "--force-overwrite",
        type=int,
        default=1,
        help="Delete the output directory if it exists already.",
    )

    parser.add_argument(
        "--policy-repo-id",
        type=str,
        default=None,
        help="Name of hugging face repositery containing a pretrained policy (e.g. `lerobot/diffusion_pusht` for https://huggingface.co/lerobot/diffusion_pusht).",
    )
    parser.add_argument(
        "--policy-ckpt-path",
        type=str,
        default=None,
        help="Name of hugging face repositery containing a pretrained policy (e.g. `lerobot/diffusion_pusht` for https://huggingface.co/lerobot/diffusion_pusht).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size loaded by DataLoader.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of processes of Dataloader for loading the data.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="local",
        help=(
            "Mode of viewing between 'local' or 'distant'. "
            "'local' requires data to be on a local machine. It spawns a viewer to visualize the data locally. "
            "'distant' creates a server on the distant machine where the data is stored. "
            "Visualize the data by connecting to the server with `rerun ws://localhost:PORT` on the local machine."
        ),
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=9090,
        help="Web port for rerun.io when `--mode distant` is set.",
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=9087,
        help="Web socket port for rerun.io when `--mode distant` is set.",
    )
    parser.add_argument(
        "--save",
        type=int,
        default=0,
        help=(
            "Save a .rrd file in the directory provided by `--output-dir`. "
            "It also deactivates the spawning of a viewer. "
            "Visualize the data by running `rerun path/to/file.rrd` on your local machine."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory path to write a .rrd file when `--save 1` is set.",
    )

    parser.add_argument(
        "--root",
        type=str,
        help="Root directory for a dataset stored on a local machine.",
    )

    args = parser.parse_args()
    visualize_dataset(**vars(args))


if __name__ == "__main__":
    main()
