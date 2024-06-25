"""This scripts demonstrates how to train Diffusion Policy on the PushT environment.

Once you have trained a model with this script, you can try to evaluate it on
examples/2_evaluate_pretrained_policy.py
"""

from pathlib import Path

import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from accelerate.utils import set_seed

set_seed(0)

# Create a directory to store the training checkpoint.
output_directory = Path("outputs/train/example_transfer_cube")
output_directory.mkdir(parents=True, exist_ok=True)

# Number of offline training steps (we'll only do offline training for this example.)
# Adjust as you prefer. 5000 steps are needed to get something worth evaluating.
training_steps = 5000
log_freq = 250


dataset = LeRobotDataset("lerobot/aloha_sim_transfer_cube_human")


# Set up the the policy.
# Policies are initialized with a configuration class, in this case `DiffusionConfig`.
# For this example, no arguments need to be passed because the defaults are set up for PushT.
# If you're doing something different, you will likely need to change at least some of the defaults.
cfg = ACTConfig()
policy = ACTPolicy(cfg, dataset_stats=dataset.stats)

from accelerate import Accelerator

accelerator = Accelerator(gradient_accumulation_steps=2)

device = accelerator.device
policy.to(device)
policy.train()

optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

# Create dataloader for offline training.
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=4,
    batch_size=64,
    shuffle=True,
    pin_memory=device != torch.device("cpu"),
    drop_last=True,
)

# Run training loop.
step = 0
done = False

policy, optimizer, dataloader = accelerator.prepare(
    policy, optimizer, dataloader
)

while not done:
    with accelerator.accumulate(policy):
        for batch in dataloader:
            output_dict = policy.forward(batch)
            loss = output_dict["loss"]
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
            step += 1
            if step >= training_steps:
                done = True
                break

accelerator.wait_for_everyone()
policy = accelerator.unwrap_model(policy)
policy.save_pretrained(
    output_directory,
    is_main_process=accelerator.is_main_process,
    save_function=accelerator.save,
)