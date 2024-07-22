import torch.nn as nn

from lerobot.scripts import train
from lerobot.common.policies.factory import _policy_cfg_from_hydra_cfg, get_policy_and_config_classes
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.act.modeling_act import ACTPolicy

from lerobot.common.utils.utils import get_safe_torch_device, init_hydra_config


from lerobot.scripts.eval import get_pretrained_policy_path

pretrained_policy_name_or_path = "m1b/2024_06_10_act_cs300_aloha_sim_transfer_cube_human"


finetune_cfg = {
    "pretrained_policy_name_or_path":"m1b/2024_06_10_act_cs300_aloha_sim_transfer_cube_human",
    "new_dataset_repo_id": "lerobot/aloha_sim_transfer_cube_human",
    "new_input_shapes": {
        "observation.images.top": [3, 480, 640],
        "observation.state": [14]
    },
    "new_output_shapes": {"action": [13]},
}

def make_finetuned_policy(hydra_cfg, finetune_cfg, dataset_stats=None):
    pretrained_policy_name_or_path = finetune_cfg["pretrained_policy_name_or_path"]
    new_output_shapes = finetune_cfg.get("new_output_shapes", None)
    new_input_shapes = finetune_cfg.get("new_input_shapes", None)

    policy_cls, policy_cfg_class = get_policy_and_config_classes(hydra_cfg.policy.name)

    policy_cfg = _policy_cfg_from_hydra_cfg(policy_cfg_class, hydra_cfg)
    policy = policy_cls(policy_cfg, dataset_stats)
    policy.load_state_dict(policy_cls.from_pretrained(pretrained_policy_name_or_path).state_dict())
    if new_input_shapes:
        policy.config.input_shapes = new_input_shapes
    if new_output_shapes:
        policy.config.output_shapes = new_output_shapes

    policy.normalize_inputs = Normalize(policy.config.input_shapes, policy.config.input_normalization_modes, dataset_stats)
    policy.normalize_targets = Normalize(policy.config.output_shapes, policy.config.output_normalization_modes, dataset_stats)
    policy.unnormalize_outputs = Unnormalize(policy.config.output_shapes, policy.config.output_normalization_modes, dataset_stats)

    if policy_cls == ACTPolicy:
        if policy.config.use_vae:
            if policy.model.config.use_vae:
                policy.vae_encoder_robot_state_input_proj = nn.Linear(
                    policy.config.input_shapes["observation.state"][0], policy.config.dim_model
                )
            # Projection layer for action (joint-space target) to hidden dimension.
            policy.vae_encoder_action_input_proj = nn.Linear(
                policy.config.output_shapes["action"][0], policy.config.dim_model
            )
        if new_input_shapes:
            if "observation.state" in new_input_shapes:
                policy.model.encoder_robot_state_input_proj = nn.Linear(
                    policy.config.input_shapes["observation.state"][0], policy.config.dim_model
                )
            if "observation.environment_state" in new_input_shapes:
                policy.model.encoder_env_state_input_proj = nn.Linear(
                    policy.config.input_shapes["observation.environment_state"][0], policy.config.dim_model
                )
        if new_output_shapes:
            policy.model.action_head = nn.Linear(policy.config.dim_model, new_output_shapes["action"][0])
    
    policy.to(get_safe_torch_device(hydra_cfg.device))

    return policy

if __name__ == "__main__":
    pretrained_policy_path = get_pretrained_policy_path(pretrained_policy_name_or_path)
    overrides = ["wandb.enable=false", "device=cpu"]
    hydra_cfg = init_hydra_config(pretrained_policy_path / "config.yaml", overrides=overrides)
    out_dir = "out_dir"
    job_name = "job_name"

    train.train(hydra_cfg, out_dir, job_name, finetune_cfg)