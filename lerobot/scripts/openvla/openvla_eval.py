import einops
import gymnasium as gym
import numpy as np
import torch
from pathlib import Path
import json
import imageio

from make_env import make_env
from lerobot.common.utils.utils import init_hydra_config, set_global_seed
from inference import OpenVLAInference

def rollout_single_episode(
    env: gym.Env,
    inference = None,
    render_video: bool = False,
    video_path: str = "./eval_episode.mp4",
    prompt: str = "",
) -> dict:
    """Run a single episode rollout with a given environment and policy.

    Args:
        env: The environment.
        policy: The policy model.
        render_video: Whether to render a video of the episode (default: False).
        video_path: Path to save the rendered video (default: "./eval_episode.mp4").
        prompt: The prompt to use for the policy (default: "").

    Returns:
        Dictionary containing rollout metrics: "action", "reward", "success", "done".
    """

    observation, info = env.reset(seed=84)
    observation_image = observation["pixels"]["top"]
    print("Observation image : ", observation_image)
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []
    ep_frames = []

    done = False
    while not done:
        with torch.inference_mode():
            action = inference.step(observation_image, task_description=prompt)
        action = action.squeeze()
        next_observation, reward, done, truncated, info = env.step(action)

        all_actions.append(torch.tensor(action))
        all_rewards.append(torch.tensor(reward))
        all_successes.append(torch.tensor(info.get("is_success", False)))
        all_dones.append(torch.tensor(done))

        if render_video:
            # Render the current frame and store it
            frame = env.render(mode='rgb_array')
            ep_frames.append(frame)

        observation = next_observation

    # Stack lists of tensors into tensors
    ret = {
        "action": torch.stack(all_actions),
        "reward": torch.stack(all_rewards),
        "success": torch.stack(all_successes),
        "done": torch.stack(all_dones),
    }

    # Save video if rendering is enabled
    if render_video and ep_frames:
        # Create video directory if it doesn't exist
        video_dir = Path(video_path).parent
        video_dir.mkdir(parents=True, exist_ok=True)

        # Save video using imageio
        imageio.mimwrite(video_path, ep_frames, fps=env.metadata["video.frames_per_second"])

    return ret

def main( output_dir, render_video, hydra_config_path, config_overrides):
    prompt = "In: What action should the robot take to pick up the red cube with the right arm and transfer it to the left arm? \nOut:"
    hydra_cfg = init_hydra_config(hydra_config_path, config_overrides)
    print("Initialized Hydra config.")
    
    # Set seed
    set_global_seed(hydra_cfg.seed)
    print("Set seed.")
    print("Environment", hydra_cfg.env.name, hydra_cfg.env.task)

    # Load inference model
    inference = OpenVLAInference(policy_setup = "aloha")
    print("Loaded inference model.")
    # Create environment
    env = make_env(hydra_cfg, n_envs=1)
    print("Created environment.")

    # Run a single episode rollout
    print("Running rollout...")
    rollout_metrics = rollout_single_episode(env, inference, render_video=render_video, prompt=prompt)

    # Save rollout metrics
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "rollout_metrics.json", "w") as f:
        json.dump(rollout_metrics, f)

if __name__ == "__main__":
    seed = 84
    output_dir = "outputs"
    render_video = False
    hydra_config_path = "/admin/home/marina_barannikov/projects/lerobot/lerobot/configs/default.yaml"
    config_overrides = ["env=aloha", "dataset_repo_id=lerobot/aloha_sim_transfer_cube_human", "env.task=AlohaTransferCube-v0"]

    main(output_dir, render_video, hydra_config_path, config_overrides)
