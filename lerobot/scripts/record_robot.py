import argparse
import concurrent.futures
import json
import shutil
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import tqdm
from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes, get_default_encoding
from lerobot.common.datasets.utils import calculate_episode_data_index, create_branch
from lerobot.common.datasets.video_utils import encode_video_frames
from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.scripts.control_sim_robot import create_rl_hf_dataset, save_image, say
from lerobot.scripts.push_dataset_to_hub import (
    push_dataset_card_to_hub,
    push_meta_data_to_hub,
    push_videos_to_hub,
    save_meta_data,
)
from pynput import keyboard
import gym_lowcostrobot

# Shared state variables
exit_early = False
rerecord_episode = False
stop_recording = False


def on_press(key):
    global exit_early, rerecord_episode, stop_recording
    try:
        if key == keyboard.Key.right:
            print("Right arrow key pressed. Exiting loop...")
            exit_early = True
        elif key == keyboard.Key.left:
            print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
            rerecord_episode = True
            exit_early = True
        elif key == keyboard.Key.esc:
            print("Escape key pressed. Stopping data recording...")
            stop_recording = True
            exit_early = True
    except Exception as e:
        print(f"Error handling key press: {e}")


def do_sim(args):
    global exit_early, rerecord_episode, stop_recording
    repo_id = args.repo_id

    env = gym.make(args.env_name, render_mode="human")

    offsets = [0, -0.5, -0.5, 0, -0.5, 0]
    offsets = np.array(offsets) * np.pi
    counts_to_radians = np.pi * 2.0 / 4096.0
    # get the start pos from .cache/calibration directory in your local lerobot
    start_pos = [1965, 3130, 912, 1970, 2075, 1956]
    axis_directions = [-1, -1, 1, -1, -1, -1]
    joint_commands = np.array((0, 0, 0, 0, 0, 0))
    leader_arm = DynamixelMotorsBus(
        port=args.device,
        motors={
            # name: (index, model)
            "shoulder_pan": (1, "xl330-m077"),
            "shoulder_lift": (2, "xl330-m077"),
            "elbow_flex": (3, "xl330-m077"),
            "wrist_flex": (4, "xl330-m077"),
            "wrist_roll": (5, "xl330-m077"),
            "gripper": (6, "xl330-m077"),
        },
    )

    if not leader_arm.is_connected:
        leader_arm.connect()

    local_dir = Path("data") / args.repo_id
    if local_dir.exists() and args.force_override:
        shutil.rmtree(local_dir)

    episodes_dir = local_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    videos_dir = local_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Logic to resume data recording
    rec_info_path = episodes_dir / "data_recording_info.json"
    if rec_info_path.exists():
        with open(rec_info_path) as f:
            rec_info = json.load(f)
        episode_index = rec_info["last_episode_index"] + 1
    else:
        episode_index = 0

    num_episodes = args.num_episodes
    fps = args.fps
    num_image_writers = args.num_image_writers
    futures = []

    state_keys = {
        "observation.state": "arm_qpos",
        "observation.velocity": "arm_qvel",
    }  # Adjust based on your environment
    image_keys = ["image_front", "image_top"]  # Adjust based on your environment's image keys

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_image_writers) as executor:
        # print(f"Running in thread: {threading.current_thread().name}")
        while episode_index < num_episodes:
            say(f"Recording episode {episode_index}")
            ep_dict = {"action": [], "next.reward": []}
            for key in state_keys:
                ep_dict[key] = []
            frame_index = 0
            start_episode_t = time.perf_counter()

            # Seed for reproducibility
            seed = np.random.randint(0, 1e5)
            observation, info = env.reset(seed=seed)
            # Use ThreadPoolExecutor to save images asynchronously

            while True:
                real_positions = np.array(leader_arm.read("Present_Position"))
                joint_commands = axis_directions * (real_positions - start_pos) * counts_to_radians + offsets

                # print(f"Running in thread: {threading.current_thread().name}")
                for key in image_keys:
                    str_key = key if key.startswith("observation.images.") else "observation.images." + key
                    futures += [
                        executor.submit(
                            save_image, observation[key], str_key, frame_index, episode_index, videos_dir
                        )
                    ]

                for key, obs_key in state_keys.items():
                    ep_dict[key].append(torch.from_numpy(observation[obs_key]))

                # Advance the sim environment
                if len(joint_commands.shape) == 1:
                    joint_commands = np.expand_dims(joint_commands, 0)
                observation, reward, _, _, info = env.step(joint_commands)
                ep_dict["action"].append(torch.from_numpy(joint_commands))
                ep_dict["next.reward"].append(torch.tensor(reward))

                frame_index += 1

                timestamp = time.perf_counter() - start_episode_t
                if exit_early:
                    exit_early = False
                    break

            num_frames = frame_index
            for key in image_keys:
                if not key.startswith("observation.images."):
                    key = "observation.images." + key

                tmp_imgs_dir = videos_dir / f"{key}_episode_{episode_index:06d}"
                fname = f"{key}_episode_{episode_index:06d}.mp4"
                video_path = local_dir / "videos" / fname
                if video_path.exists():
                    video_path.unlink()
                # Store the reference to the video frame, even tho the videos are not yet encoded
                ep_dict[key] = []
                for i in range(num_frames):
                    ep_dict[key].append({"path": f"videos/{fname}", "timestamp": i / fps})

            for key in state_keys:
                ep_dict[key] = torch.vstack(ep_dict[key]) * 180.0 / np.pi
            ep_dict["action"] = torch.vstack(ep_dict["action"]) * 180.0 / np.pi
            ep_dict["next.reward"] = torch.stack(ep_dict["next.reward"])

            ep_dict["seed"] = torch.tensor([seed] * num_frames)
            ep_dict["episode_index"] = torch.tensor([episode_index] * num_frames)
            ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
            ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / fps

            done = torch.zeros(num_frames, dtype=torch.bool)
            done[-1] = True
            ep_dict["next.done"] = done

            ep_path = episodes_dir / f"episode_{episode_index}.pth"
            print("Saving episode dictionary...")
            torch.save(ep_dict, ep_path)

            rec_info = {
                "last_episode_index": episode_index,
            }
            with open(rec_info_path, "w") as f:
                json.dump(rec_info, f)

            is_last_episode = stop_recording or (episode_index == (num_episodes - 1))

            # Skip updating episode index which forces re-recording episode
            if rerecord_episode:
                rerecord_episode = False
                continue

            episode_index += 1
            if is_last_episode:
                print("Done recording")
                say("Done recording", blocking=True)

                print("Waiting for threads writing the images on disk to terminate...")
                for _ in tqdm.tqdm(
                    concurrent.futures.as_completed(futures), total=len(futures), desc="Writting images"
                ):
                    pass
            else:
                print("Waiting for two seconds before starting the next recording session.....")
                busy_wait(2)

    num_episodes = episode_index

    say("Encoding videos")
    # Use ffmpeg to convert frames stored as png into mp4 videos
    for episode_index in tqdm.tqdm(range(num_episodes)):
        for key in image_keys:
            if not key.startswith("observation.images."):
                key = "observation.images." + key

            tmp_imgs_dir = videos_dir / f"{key}_episode_{episode_index:06d}"
            fname = f"{key}_episode_{episode_index:06d}.mp4"
            video_path = local_dir / "videos" / fname
            if video_path.exists():
                # Skip if video is already encoded. Could be the case when resuming data recording.
                continue
            # note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
            # since video encoding with ffmpeg is already using multithreading.
            encode_video_frames(tmp_imgs_dir, video_path, fps, overwrite=True)
            shutil.rmtree(tmp_imgs_dir)

    ep_dicts = []
    for episode_index in tqdm.tqdm(range(num_episodes)):
        ep_path = episodes_dir / f"episode_{episode_index}.pth"
        ep_dict = torch.load(ep_path)
        ep_dicts.append(ep_dict)
    data_dict = concatenate_episodes(ep_dicts)

    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames)

    hf_dataset = create_rl_hf_dataset(data_dict)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "video": True,
    }
    info["encoding"] = get_default_encoding()

    lerobot_dataset = LeRobotDataset.from_preloaded(
        repo_id=repo_id,
        hf_dataset=hf_dataset,
        episode_data_index=episode_data_index,
        info=info,
        videos_dir=videos_dir,
    )

    if args.run_compute_stats:
        print("Computing dataset statistics")
        say("Computing dataset statistics")
        stats = compute_stats(lerobot_dataset)
        lerobot_dataset.stats = stats
    else:
        stats = {}
        print("Skipping computation of the dataset statistics")

    hf_dataset = hf_dataset.with_format(None)  # to remove transforms that cant be saved
    hf_dataset.save_to_disk(str(local_dir / "train"))

    meta_data_dir = local_dir / "meta_data"
    save_meta_data(info, stats, episode_data_index, meta_data_dir)

    if args.push_to_hub:
        hf_dataset.push_to_hub(repo_id, revision="main")
        push_meta_data_to_hub(repo_id, meta_data_dir, revision="main")
        push_dataset_card_to_hub(repo_id, revision="main", tags=["lerobot"])
        push_videos_to_hub(repo_id, videos_dir, revision="main")
        create_branch(repo_id, repo_type="dataset", branch=CODEBASE_VERSION)

    say("Exiting")
    return lerobot_dataset


if __name__ == "__main__":
    import threading

    parser = argparse.ArgumentParser(
        description="Teleoperate and record data from low-cost robot simulation."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="/dev/ttyACM0",
        help="Port name (e.g., COM1, /dev/ttyUSB0, /dev/tty.usbserial-*)",
    )
    parser.add_argument(
        "--env-name", type=str, default="PushCubeLoop-v0", help="Specify the gym-lowcost robot env to test."
    )
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for recording")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="your-username/your-dataset-name",
        help="Hugging Face Hub repository ID",
    )
    parser.add_argument(
        "--force-override", action="store_true", help="Override existing data in the local directory"
    )
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes to record")
    parser.add_argument("--num-image-writers", type=int, default=4, help="Number of threads for image saving")
    parser.add_argument("--push-to-hub", action="store_true", help="Push the dataset to the Hugging Face Hub")
    parser.add_argument("--run-compute-stats", action="store_true", help="Compute dataset statistics")
    args = parser.parse_args()
    print(args)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    print(f"Running in thread: {threading.current_thread().name}")
    # Start the main simulation in a separate thread
    main_thread = threading.Thread(target=do_sim, args=(args,))
    main_thread.start()

    # Wait for the main simulation to finish
    main_thread.join()

    # Stop the keyboard listener
    listener.stop()
