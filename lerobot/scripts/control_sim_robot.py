"""
Utilities to control a robot in simulation.

Useful to record a dataset, replay a recorded episode and record an evaluation dataset.

Examples of usage:


- Unlimited teleoperation at a limited frequency of 30 Hz, to simulate data recording frequency.
  You can modify this value depending on how fast your simulation can run:
```bash
python lerobot/scripts/control_robot.py teleoperate \
    --fps 30 \
    --robot-path lerobot/configs/robot/your_robot_config.yaml \
    --sim-config lerobot/configs/env/your_sim_config.yaml
```

- Record one episode in order to test replay:
```bash
python lerobot/scripts/control_sim_robot.py record \
    --robot-path lerobot/configs/robot/your_robot_config.yaml \
    --sim-config lerobot/configs/env/your_sim_config.yaml \
    --fps 30 \
    --root tmp/data \
    --repo-id $USER/robot_sim_test \
    --num-episodes 1 \
    --run-compute-stats 0
```

- Visualize dataset:
```bash
python lerobot/scripts/visualize_dataset.py \
    --root tmp/data \
    --repo-id $USER/robot_sim_test \
    --episode-index 0
```

- Replay this test episode:
```bash
python lerobot/scripts/control_sim_robot.py replay \
    --sim-config lerobot/configs/env/your_sim_config.yaml \
    --fps 30 \
    --root tmp/data \
    --repo-id $USER/koch_test \
    --episodes 0
```

- Record a full dataset in order to train a policy,
30 seconds of recording for each episode, and 10 seconds to reset the environment in between episodes:
```bash
python lerobot/scripts/control_sim_robot.py record \
    --robot-path lerobot/configs/robot/your_robot_config.yaml \
    --sim-config lerobot/configs/env/your_sim_config.yaml \
    --fps 30 \
    --root data \
    --repo-id $USER/robot_sim_test \
    --num-episodes 50 \
    --episode-time-s 30 \
    --reset-time-s 10
```

**NOTE**: You can use your keyboard to control data recording flow.
- Tap right arrow key '->' to early exit while recording an episode and go to resseting the environment.
- Tap right arrow key '->' to early exit while resetting the environment and got to recording the next episode.
- Tap left arrow key '<-' to early exit and re-record the current episode.
- Tap escape key 'esc' to stop the data recording.
This might require a sudo permission to allow your terminal to monitor keyboard events.

**NOTE**: You can resume/continue data recording by running the same data recording command twice.
To avoid resuming by deleting the dataset, use `--force-override 1`.

"""

import argparse
import concurrent.futures
import json
import logging
import multiprocessing.process
import os
import platform
import shutil
import time
import traceback
import warnings
warnings.simplefilter("ignore", UserWarning)
from functools import cache
from pathlib import Path
import gymnasium as gym
import multiprocessing 

import cv2
import torch
import numpy as np
import tqdm
from PIL import Image
from datasets import Dataset, Features, Sequence, Value

# from safetensors.torch import load_file, save_file
from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.common.datasets.video_utils import VideoFrame, encode_video_frames
from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes, get_default_encoding
from lerobot.common.datasets.utils import calculate_episode_data_index, create_branch, hf_transform_to_torch
from lerobot.common.datasets.video_utils import encode_video_frames
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.envs.factory import make_env
from lerobot.common.utils.utils import init_hydra_config, init_logging
from lerobot.scripts.push_dataset_to_hub import (
    push_dataset_card_to_hub,
    push_meta_data_to_hub,
    push_videos_to_hub,
    save_meta_data,
)

########################################################################################
# Utilities
########################################################################################
def say(text, blocking=False):
    # Check if mac, linux, or windows.
    if platform.system() == "Darwin":
        cmd = f'say "{text}"'
    elif platform.system() == "Linux":
        cmd = f'spd-say "{text}"'
    elif platform.system() == "Windows":
        cmd = (
            'PowerShell -Command "Add-Type -AssemblyName System.Speech; '
            f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')\""
        )

    if not blocking and platform.system() in ["Darwin", "Linux"]:
        # TODO(rcadene): Make it work for Windows
        # Use the ampersand to run command in the background
        cmd += " &"

    os.system(cmd)


def save_image(img_arr, key, frame_index, episode_index, videos_dir):
    img = Image.fromarray(img_arr)
    path = videos_dir / f"{key}_episode_{episode_index:06d}" / f"frame_{frame_index:06d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), quality=100)

def show_image_observations(observation_queue:multiprocessing.Queue):
    keys = None
    while True:
        observations = observation_queue.get()
        images = []
        if keys is None: keys = [k for k in observations if 'image' in k]
        for key in keys:
            images.append(observations[key].squeeze(0))
        cat_image = np.concatenate(images, 1)
        cv2.imshow('observations', cv2.cvtColor(cat_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)


def none_or_int(value):
    if value == "None":
        return None
    return int(value)


@cache
def is_headless():
    """Detects if python is running without a monitor."""
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print() 
        return True

def init_read_leader(robot, fps, **kwargs):
    axis_directions = kwargs.get('axis_directions', [1])
    offsets = kwargs.get('offsets', [0])
    command_queue = multiprocessing.Queue(1000)
    read_leader = multiprocessing.Process(target=read_commands_from_leader, args=(robot, command_queue, fps, axis_directions, offsets))
    return read_leader, command_queue

def read_commands_from_leader(robot: Robot, queue: multiprocessing.Queue, fps: int, axis_directions: list, offsets: list, stop_flag=None):
    if not robot.is_connected:
        robot.connect()

    # Constants necessary for transforming the joint pos of the real robot to the sim
    # depending on the robot discription used in that sim.
    start_pos = np.array(robot.leader_arms.main.calibration['start_pos'])
    axis_directions = np.array(axis_directions)
    offsets = np.array(offsets) * np.pi
    counts_to_radians = np.pi * 2. / 4096

    if stop_flag is None:
        stop_flag = multiprocessing.Value('b', False)

    #TODO(michel_aractingi): temp fix to disable calibration while reading from the leader arms
    # different calculation for joint commands would be needed
    robot.leader_arms.main.calibration = None 
    while True:
        #with stop_flag.get_lock():  
        #    stop_flag_value = stop_flag.value

        start_loop_t = time.perf_counter()
        #if not stop_flag_value:
        real_positions = np.array(robot.leader_arms.main.read('Present_Position'))
        joint_commands = axis_directions * (real_positions - start_pos) * counts_to_radians + offsets
        queue.put(joint_commands)
        if fps is not None:
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)
        #else:
            #queue.get() #TODO (michel_aractingi): remove elements from queue in case get_lock is delayed 
            #print('here!!!')
            #busy_wait(0.01)
        
def create_rl_hf_dataset(data_dict):
    features = {}

    keys = [key for key in data_dict if "observation.images." in key]
    for key in keys:
        features[key] = VideoFrame()

    features["observation.state"] = Sequence(
        length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
    )
    if "observation.velocity" in data_dict:
        features["observation.velocity"] = Sequence(
            length=data_dict["observation.velocity"].shape[1], feature=Value(dtype="float32", id=None)
        )
    if "observation.effort" in data_dict:
        features["observation.effort"] = Sequence(
            length=data_dict["observation.effort"].shape[1], feature=Value(dtype="float32", id=None)
        )
    features["action"] = Sequence(
        length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["reward"] = Value(dtype="float32", id=None)

    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"] = Value(dtype="int64", id=None)
    features["timestamp"] = Value(dtype="float32", id=None)
    features["next.done"] = Value(dtype="bool", id=None)
    features["index"] = Value(dtype="int64", id=None)
    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset


########################################################################################
# Control modes
########################################################################################


def teleoperate(env, robot: Robot, teleop_time_s=None, teleop_method='arm', **kwargs):    
    env = env()
    env.reset()
    
    read_leader, command_queue = init_read_leader(robot, **kwargs)
    start_teleop_t = time.perf_counter() 
    read_leader.start()
    while True:
        action = command_queue.get()
        env.step(np.expand_dims(action, 0))
        if teleop_time_s is not None and time.perf_counter() - start_teleop_t > teleop_time_s:
            read_leader.terminate()
            command_queue.close()
            print("Teleoperation processes finished.")
            break

def record(
    env, 
    robot: Robot,
    fps: int | None = None,
    root="data",
    repo_id="lerobot/debug",
    episode_time_s=30,
    num_episodes=50,
    video=True,
    run_compute_stats=True,
    push_to_hub=True,
    tags=None,
    num_image_writers_per_camera=4,
    force_override=False,
    visualize_images=0,
    teleop_method='arm',
    **kwargs
):

    local_dir = Path(root) / repo_id
    if local_dir.exists() and force_override:
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

    if is_headless():
        logging.warning(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )

    final_joint_positions = np.array([-0.00306796,  0.71811652,  1.41732051,  -0.15493206,   0.09203885, -0.75471855])
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.
    set_final_joint_positions = False
    exit_early = False
    rerecord_episode = False
    stop_recording = False
    # Only import pynput if not in a headless environment
    if not is_headless():
        from pynput import keyboard

        def on_press(key):
            nonlocal exit_early, rerecord_episode, stop_recording, set_final_joint_positions
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
                elif key == keyboard.Key.down:
                    print("Down arrow key pressed. Setting end joint positions...")
                    set_final_joint_positions = True

                if teleop_method == 'keyboard':
                    if np.isnan(np.linalg.norm(teleop_action)):
                        # Assign good initial action
                        # FIXME: This is robot-specific (Low-Cost Robot Arm)
                        teleop_action[0][0] = 0.0
                        teleop_action[0][1] = 0.14
                        teleop_action[0][2] = 0.17
                        teleop_action[0][3] = 0.0

                    assert np.isfinite(np.linalg.norm(teleop_action))
                    if key == keyboard.KeyCode.from_char('w'):
                        teleop_action[0][1] += 0.1  # Move forward
                    elif key == keyboard.KeyCode.from_char('s'):
                        teleop_action[0][1] -= 0.1  # Move backward
                    elif key == keyboard.KeyCode.from_char('a'):
                        teleop_action[0][0] -= 0.1  # Move left
                    elif key == keyboard.KeyCode.from_char('d'):
                        teleop_action[0][0] += 0.1  # Move right
                    elif key == keyboard.KeyCode.from_char('q'):
                        teleop_action[0][2] += 0.1  # Move up
                    elif key == keyboard.KeyCode.from_char('e'):
                        teleop_action[0][2] -= 0.1  # Move down
                    elif key == keyboard.KeyCode.from_char('r'):
                        teleop_action[0][3] += 0.1  # Open gripper
                    elif key == keyboard.KeyCode.from_char('f'):
                        teleop_action[0][3] -= 0.1  # Close gripper
                    
                    command_queue.put(teleop_action.copy())

            except Exception as e:
                print(f"Error handling key press: {e}")

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
    
    # create env
    env = env()

    # Save images using threads to reach high fps (30 and more)
    # Using `with` to exist smoothly if an execption is raised.
    futures = []
    num_image_writers = num_image_writers_per_camera * 2 ###############
    num_image_writers = max(num_image_writers, 1)

    if teleop_method == 'arm':
        print('Using leader arm to teleoperate the robot')
        read_leader, command_queue = init_read_leader(robot, fps, **kwargs)
    else:
        print('Using keyboard to teleoperate the robot')
        print('Press "w" to move forward, "s" to move backward, "a" to move left, "d" to move right, "q" to move up, "e" to move down, "r" to open gripper, "f" to close gripper')
        command_queue = multiprocessing.Queue(1000)

    if not is_headless() and visualize_images:
        observations_queue = multiprocessing.Queue(1000)
        show_images = multiprocessing.Process(target=show_image_observations, args=(observations_queue, ))
        show_images.start()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_image_writers) as executor:
        # Start recording all episodes
        # start reading from leader, disable stop flag in leader process
        while episode_index < num_episodes:
            logging.info(f"Recording episode {episode_index}")
            say(f"Recording episode {episode_index}")
            ep_dict = {'action':[], 'observation.state': [], 'reward':[]}
            frame_index = 0
            timestamp = 0
            start_episode_t = time.perf_counter()
            observation, info = env.reset()
            set_final_joint_positions = False
            #with stop_reading_leader.get_lock(): 
                #stop_reading_leader.Value = 0
            if teleop_method == 'arm':
                read_leader.start()
            else :
                teleop_action = np.zeros(env.action_space.shape)
            while timestamp < episode_time_s:
                try:
                    action = command_queue.get(timeout=0.1)
                except multiprocessing.queues.Empty:
                    action = np.zeros(env.action_space.shape)
                if set_final_joint_positions:
                    action = final_joint_positions
                print('Final position was set ? ', set_final_joint_positions)
                image_keys = [key for key in observation if "image" in key]
                state_keys = [key for key in observation if "image" not in key]
                for key in image_keys:
                    str_key = key if key.startswith('observation.images.') else 'observation.images.' + key
                    futures += [
                        executor.submit(
                            save_image, observation[key].squeeze(0), str_key, frame_index, episode_index, videos_dir)
                    ]

                if not is_headless() and visualize_images:
                    observations_queue.put(observation)
          
                state_obs = []
                for key in state_keys:
                    state_obs.append(torch.from_numpy(observation[key]))
                ep_dict['observation.state'].append(torch.hstack(state_obs))

                # Advance the sim environment
                if len(action.shape) == 1:
                    action = np.expand_dims(action, 0)
                observation, reward, _, _ , info = env.step(action)
                ep_dict['action'].append(torch.from_numpy(action))
                ep_dict['reward'].append(torch.tensor(reward))

                frame_index += 1

                timestamp = time.perf_counter() - start_episode_t

                if exit_early:
                    exit_early = False
                    break

            # enable stop reading leader flag
            #with stop_reading_leader.get_lock(): 
                #stop_reading_leader.Value = 1
            # TODO (michel_aractinig): temp fix until I figure out the problem with shared memory
            # stop_reading_leader is blocking
            command_queue.close()
            if teleop_method == 'arm':
                read_leader.terminate() 
                read_leader, command_queue = init_read_leader(robot, fps, **kwargs)
            else:
                command_queue = multiprocessing.Queue(1000)    

            timestamp = 0

            # During env reset we save the data and encode the videos
            num_frames = frame_index

            for key in image_keys:
                if not key.startswith('observation.images.'):
                    key = 'observation.images.' + key

                if video:
                    tmp_imgs_dir = videos_dir / f"{key}_episode_{episode_index:06d}"
                    fname = f"{key}_episode_{episode_index:06d}.mp4"
                    video_path = local_dir / "videos" / fname
                    if video_path.exists():
                        video_path.unlink()
                    # Store the reference to the video frame, even tho the videos are not yet encoded
                    ep_dict[key] = []
                    for i in range(num_frames):
                        ep_dict[key].append({"path": f"videos/{fname}", "timestamp": i / fps})

                else:
                    imgs_dir = videos_dir / f"{key}_episode_{episode_index:06d}"
                    ep_dict[key] = []
                    for i in range(num_frames):
                        img_path = imgs_dir / f"frame_{i:06d}.png"
                        ep_dict[key].append({"path": str(img_path)})

            ep_dict['observation.state'] = torch.vstack(ep_dict['observation.state'])
            ep_dict['action'] = torch.vstack(ep_dict['action'])
            ep_dict['reward'] = torch.stack(ep_dict['reward'])

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
                logging.info("Done recording")
                say("Done recording", blocking=True)

                logging.info("Waiting for threads writing the images on disk to terminate...")
                for _ in tqdm.tqdm(
                    concurrent.futures.as_completed(futures), total=len(futures), desc="Writting images"
                ):
                    pass
                if not is_headless() and visualize_images:
                    show_images.terminate()
                    observations_queue.close()
                break
            else:
                print('Waiting for ten seconds before starting the next recording session.....')


    num_episodes = episode_index

    if video:
        logging.info("Encoding videos")
        say("Encoding videos")
        # Use ffmpeg to convert frames stored as png into mp4 videos
        for episode_index in tqdm.tqdm(range(num_episodes)):
            for key in image_keys:
                if not key.startswith('observation.images.'):
                    key = 'observation.images.' + key

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

    logging.info("Concatenating episodes")
    ep_dicts = []
    for episode_index in tqdm.tqdm(range(num_episodes)):
        ep_path = episodes_dir / f"episode_{episode_index}.pth"
        ep_dict = torch.load(ep_path)
        ep_dicts.append(ep_dict)
    data_dict = concatenate_episodes(ep_dicts)

    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)

    hf_dataset = create_rl_hf_dataset(data_dict)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "video": video,
    }
    if video:
        info["encoding"] = get_default_encoding()

    lerobot_dataset = LeRobotDataset.from_preloaded(
        repo_id=repo_id,
        hf_dataset=hf_dataset,
        episode_data_index=episode_data_index,
        info=info,
        videos_dir=videos_dir,
    )
    if run_compute_stats:
        logging.info("Computing dataset statistics")
        say("Computing dataset statistics")
        stats = compute_stats(lerobot_dataset)
        lerobot_dataset.stats = stats
    else:
        stats = {}
        logging.info("Skipping computation of the dataset statistics")

    hf_dataset = hf_dataset.with_format(None)  # to remove transforms that cant be saved
    hf_dataset.save_to_disk(str(local_dir / "train"))

    meta_data_dir = local_dir / "meta_data"
    save_meta_data(info, stats, episode_data_index, meta_data_dir)

    if push_to_hub:
        hf_dataset.push_to_hub(repo_id, revision="main")
        push_meta_data_to_hub(repo_id, meta_data_dir, revision="main")
        push_dataset_card_to_hub(repo_id, revision="main", tags=tags)
        if video:
            push_videos_to_hub(repo_id, videos_dir, revision="main")
        create_branch(repo_id, repo_type="dataset", branch=CODEBASE_VERSION)

    logging.info("Exiting")
    say("Exiting")
    return lerobot_dataset


def replay(env, episodes: list, fps: int | None = None, root="data", repo_id="lerobot/debug"):

    env = env()
    local_dir = Path(root) / repo_id
    if not local_dir.exists():
        raise ValueError(local_dir)

    dataset = LeRobotDataset(repo_id, root=root)
    items = dataset.hf_dataset.select_columns("action")
    for episode in episodes:
        env.reset()
        from_idx = dataset.episode_data_index["from"][episode].item()
        to_idx = dataset.episode_data_index["to"][episode].item()
    
        logging.info("Replaying episode")
        say("Replaying episode", blocking=True)
        for idx in range(from_idx, to_idx):
            start_episode_t = time.perf_counter()
    
            action = items[idx]["action"]
    
            env.step(action.unsqueeze(0).numpy())
    
            dt_s = time.perf_counter() - start_episode_t
            busy_wait(1 / fps - dt_s)

        # wait before playing next episode
        busy_wait(5)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Set common options for all the subparsers
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--robot-path",
        type=str,
        default="lerobot/configs/robot/koch.yaml",
        help="Path to robot yaml file used to instantiate the robot using `make_robot` factory function.",
    )
    
    base_parser.add_argument(
        "--sim-config",
        help="Path to a yaml config you want to use for initializing a sim environment based on gym ",
        )

    parser_teleop = subparsers.add_parser("teleoperate", parents=[base_parser])
    parser_teleop.add_argument(
        "--fps", type=none_or_int, default=None, help="Frames per second (set to None to disable)"
    )
    parser_teleop.add_argument(
        "--teleop-method", type=str, default='arm', choices=["keyboard", "arm"], help="Method to teleoperate the robot in the environment. Options are 'arm' or 'keyboard'."
    )

    parser_record = subparsers.add_parser("record", parents=[base_parser])
    parser_record.add_argument(
        "--fps", type=none_or_int, default=None, help="Frames per second (set to None to disable)"
    )
    parser_record.add_argument(
        "--teleop-method", type=str, default='arm', choices=["keyboard", "arm"], help="Method to teleoperate the robot in the environment. Options are 'arm' or 'keyboard'."
    )
    parser_record.add_argument(
        "--root",
        type=Path,
        default="data",
        help="Root directory where the dataset will be stored locally at '{root}/{repo_id}' (e.g. 'data/hf_username/dataset_name').",
    )
    parser_record.add_argument(
        "--repo-id",
        type=str,
        default="lerobot/test",
        help="Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).",
    )
    parser_record.add_argument(
        "--episode-time-s",
        type=int,
        default=60,
        help="Number of seconds for data recording for each episode.",
    )
    parser_record.add_argument(
        "--reset-time-s",
        type=int,
        default=60,
        help="Number of seconds for resetting the environment after each episode.",
    )
    parser_record.add_argument("--num-episodes", type=int, default=50, help="Number of episodes to record.")
    parser_record.add_argument(
        "--run-compute-stats",
        type=int,
        default=1,
        help="By default, run the computation of the data statistics at the end of data collection. Compute intensive and not required to just replay an episode.",
    )
    parser_record.add_argument(
        "--push-to-hub",
        type=int,
        default=1,
        help="Upload dataset to Hugging Face hub.",
    )
    parser_record.add_argument(
        "--tags",
        type=str,
        nargs="*",
        help="Add tags to your dataset on the hub.",
    )
    parser_record.add_argument(
        "--num-image-writers-per-camera",
        type=int,
        default=4,
        help=(
            "Number of threads writing the frames as png images on disk, per camera. "
            "Too much threads might cause unstable teleoperation fps due to main thread being blocked. "
            "Not enough threads might cause low camera fps."
        ),
    )
    parser_record.add_argument(
        "--force-override",
        type=int,
        default=0,
        help="By default, data recording is resumed. When set to 1, delete the local directory and start data recording from scratch.",
    )
    parser_record.add_argument(
        "--visualize-images",
        type=int,
        default=0,
        help="Visualize image observations with opencv.",
    )

    parser_replay = subparsers.add_parser("replay", parents=[base_parser])
    parser_replay.add_argument(
        "--fps", type=none_or_int, default=None, help="Frames per second (set to None to disable)"
    )
    parser_replay.add_argument(
        "--root",
        type=Path,
        default="data",
        help="Root directory where the dataset will be stored locally at '{root}/{repo_id}' (e.g. 'data/hf_username/dataset_name').",
    )
    parser_replay.add_argument(
        "--repo-id",
        type=str,
        default="lerobot/test",
        help="Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).",
    )
    parser_replay.add_argument("--episodes", nargs='+', type=int, default=[0], help="Indices of the episodes to replay.")

    args = parser.parse_args()

    init_logging()

    control_mode = args.mode
    robot_path = args.robot_path
    env_config_path = args.sim_config
    kwargs = vars(args)
    del kwargs["mode"]
    del kwargs["robot_path"]
    del kwargs["sim_config"]

    # make gym env
    env_cfg = init_hydra_config(env_config_path)
    env_fn = lambda: make_env(env_cfg, n_envs=1)
    
    robot = None
    if control_mode != 'replay':
        # make robot
        robot_overrides = ['~cameras', '~follower_arms']
        robot_cfg = init_hydra_config(robot_path, robot_overrides)
        robot = make_robot(robot_cfg)
    
        kwargs.update(env_cfg.calibration)

    if control_mode == "teleoperate":
        teleoperate(env_fn, robot, **kwargs)

    elif control_mode == "record":
        record(env_fn, robot, **kwargs)

    elif control_mode == "replay":
        replay(env_fn, **kwargs)

    else:
        raise ValueError(f"Invalid control mode: '{control_mode}', only valid modes are teleoperate, record and replay." )

    if robot and robot.is_connected:
        # Disconnect manually to avoid a "Core dump" during process
        # termination due to camera threads not properly exiting.
        robot.disconnect()
