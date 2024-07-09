from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers import LlamaTokenizerFast

from PIL import Image

import json
from pathlib import Path
import torch
from collections import deque
import numpy as np

from action_ensemble import ActionEnsembler

def get_from_url(url):
    import requests
    from io import BytesIO
    return Image.open(BytesIO(requests.get(url).content))

device = "cuda:0"
vla_path = "/admin/home/marina_barannikov/projects/lerobot/lerobot/scripts/openvla/runs/step-4000"

def get_single_inference():
    # Load Processor & VLA
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        vla_path, 
        attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        local_files_only=True, 
        trust_remote_code=True
    ).to(device)

    # Grab image input & format prompt
    image: Image.Image = get_from_url("https://datasets-server.huggingface.co/assets/lerobot/aloha_sim_transfer_cube_scripted_image/--/0fb0e9c38e8e876678502996a0fe6b123f59c57a/--/default/train/3/observation.images.top/image.jpg?Expires=1720519339&Signature=zwrq4YfonSOB3oahBOUlm0GnJdBOlWkWFk0YYJmruBJDk47q3CRKGprj0akVt3xqI3~XhG64OqHmEwBBaWZtSrbVPXobnKJvU8EsrNBQsvHBj-XZxicJJuHVdtWDRSCTRLurnYQ1dHLjAM-twiaoS9WOBuPsKDPqWbjjm2HW2OQZGgnAsWf9nJNeB2Z0nko5xdhBGn8YJC-8wk1QpemnOROpJd352X3Q927oK~mHV4em4fpamf2sQor3O7bv078B46QdoMuwb-JggMwc84TDeouANw6j4X2wtDElqIyKf9yLFMXoz~fE75ciDD45KclUQ0m6pyYYcgu8byhuUeDHhg__&Key-Pair-Id=K3EI6M078Z3AC3")
    prompt = "In: What action should the robot take to pick up the red cube with the right arm and transfer it to the left arm? \nOut:"

    # Predict Action (7-DoF; un-normalize for BridgeData V2)
    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)

    def print_attribute_names(obj, name):
        print(f"Attributes of {name}:")
        for attribute in dir(obj):
            if not attribute.startswith("__"):
                print(attribute)

            
    # Processor Attributes

    #print_attribute_names(inputs, "inputs")
    #print("Vla config namemed modules : ", vla.named_modules)

    with open('runs/dataset_statistics.json', 'r') as f:
        dataset_stats_dict = json.load(f)

    vla.norm_stats['aloha'] = dataset_stats_dict

    action = vla.predict_action(**inputs, unnorm_key="aloha", do_sample=False)

    print(action)

get_single_inference()


class OpenVLAInference:
    def __init__(
        self,
        saved_model_path: str = "/admin/home/marina_barannikov/projects/lerobot/runs/openvlamodel",
        unnorm_key = "aloha",
        policy_setup: str = None,
        horizon: int = 2,
        pred_action_horizon: int = 1,
        exec_horizon: int = 1,
    ) -> None:
        if policy_setup == "aloha":
            unnorm_key = "aloha" if unnorm_key is None else unnorm_key
            action_ensemble = True
            action_ensemble_temp = 0.0
        
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        self.processor = AutoProcessor.from_pretrained(saved_model_path, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            saved_model_path,
            attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True, 
            local_files_only=True, 
        ).to("cuda:0")

        if policy_setup == "aloha":
            with open('runs/dataset_statistics.json', 'r') as f:
                dataset_stats_dict = json.load(f)
            self.vla.norm_stats['aloha'] = dataset_stats_dict

        self.horizon = horizon
        self.pred_action_horizon = pred_action_horizon
        self.exec_horizon = exec_horizon
        self.action_ensemble = action_ensemble
        self.action_ensemble_temp = action_ensemble_temp

        self.task = None
        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = ActionEnsembler(self.pred_action_horizon, self.action_ensemble_temp)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

    def step(
        self, image, task_description = None, *args, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        assert image.dtype == np.uint8
        self._add_image_to_history(image)

        image: Image.Image = Image.fromarray(image)
        prompt = task_description

        # predict action (14-dof; un-normalize for aloha)
        inputs = self.processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
        raw_actions = self.vla.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)[None]
        print(f"*** raw actions {raw_actions} ***")

        if self.action_ensemble:
            raw_actions = self.action_ensembler.ensemble_action(raw_actions)[None]

        action = {}   
        action["terminate_episode"] = np.array([0.0])

        return raw_actions
