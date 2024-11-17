"""
This script demonstrates how to train ACT policy with distributed training on the Aloha environment
on the Transfer cube task, using HuggingFace accelerate.

Apart from the main installation procedure, please also make sure you have installed accelerate before running this script: `pip install accelerate`.

To launch it, you will have to use the accelerate launcher, for example:
`python -m accelerate.commands.launch examples/8_train_policy_distributed.py`. This will launch the script with default distributed parameters.
To launch on two GPUs, you can use `python -m accelerate.commands.launch  --num_processes 2 lerobot/examples/7_train_policy_distributed.py`.

Find detailed information in the documentation: `https://github.com/huggingface/accelerate`.
"""

from pathlib import Path

import torch
from accelerate import Accelerator

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy

# Create a directory to store the training checkpoint.
output_directory = Path("outputs/train/example_aloha_act_distributed")
output_directory.mkdir(parents=True, exist_ok=True)

# Number of overall offline training steps
training_steps = 5000
log_freq = 250

# The chunk size is the number of actions that the policy will predict.
chunk_size = 100

delta_timestamps = {
    "action":
    # Load the current action, the next 100 actions to be executed, because we train the policy
    # to predict the next 100 actions. Two frames differ by 1/50 seconds, which corresponds to the FPS of this dataset.
    [i / 50 for i in range(chunk_size)]
}


def train():
    # We prepare for distributed training using the Accelerator.
    accelerator = Accelerator()
    device = accelerator.device

    # Set up the dataset.
    dataset = LeRobotDataset("lerobot/aloha_sim_transfer_cube_human_image", delta_timestamps=delta_timestamps)
    accelerator.print(f"Loaded dataset with {len(dataset)} samples.")

    # The policy is initialized with a configuration class, in this case `ACTConfig`.
    cfg = ACTConfig(chunk_size=chunk_size)
    policy = ACTPolicy(cfg, dataset_stats=dataset.stats)
    policy.train()
    num_total_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    accelerator.print(f"Policy initialized with {num_total_params} parameters.")

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-5)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=8,
        shuffle=True,
        pin_memory=device != torch.device("cpu"),
        drop_last=True,
    )

    # Prepare the policy, optimizer, and dataloader for distributed training.
    # This will wrap the policy in a DistributedDataParallel and apply torch.autocast to the forward functions.
    policy, optimizer, dataloader = accelerator.prepare(policy, optimizer, dataloader)

    policy.to(device)

    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output_dict = policy.forward(batch)

            loss = output_dict["loss"].mean()
            accelerator.backward(loss)

            optimizer.step()
            optimizer.zero_grad()

            step += 1

            if step % log_freq == 0:
                accelerator.print(f"step: {step} loss: {loss.item():.3f}")

            if step >= training_steps:
                done = True
                break

    # Unwrap the policy of its distributed training wrapper and save it.
    unwrapped_policy = accelerator.unwrap_model(policy)
    unwrapped_policy.save_pretrained(output_directory)

    accelerator.print("Finished offline training")


# We need to add a call to the training function in order to be able to use the Accelerator.
if __name__ == "__main__":
    train()
