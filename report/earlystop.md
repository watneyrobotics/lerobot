# How to Early Stop in Imitation Learning?

In traditional supervised learning, metrics like Accuracy for classification, or Mean Squared Error for regressions serve as strong indicators of model performance, allowing us to ensure the highest generalization capability. When we choose to use a validation set, it means we usually train on only 80% of the data and evaluate the model on the 20% left-out and unseen subset. We can then use those metrics to decide when to stop training.

However, in the context of robotics via imitation learning, there is no clear consensus among authors in the literature on the best metrics or practices for early stopping. Metrics like success rate can't be computed; the policy has to be evaluated directly in the environment.

## Table of Contents

- [Introduction](#introduction)
- [PushT](#pusht)
  - [Experimental Setup](#experimental-setup)
  - [Results](#results)
- [Transfer Cube](#transfer-cube)
  - [Experimental Setup](#experimental-setup-1)
  - [Results](#results-1)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction

[ACT and Aloha](arxiv.org/pdf/2304.13705)  authors Zhao et al. indicate that *"at test time, we load the policy that achieves the lowest validation loss and roll it out in the environment"*, but Stanford Robomimic authors notice *"that the best [policy early stopped on validation loss] is 50 to 100% worse than the best performing policy [when we evaluate all the checkpoints]"*.

There are multiple high-level hypotheses for the misalignment between low validation loss and high success rate:

- There is a shift in distribution between training and evaluation. This may be due to changing environments, with factors such as varying room layouts, lighting conditions, or motor temperature.
- The prediction error accumulates over time and grows as we reach the end of the predicted trajectory. This slowly puts the robot outside its training distribution.
- Through its loss function, imitation learning optimizes copying a human demonstrator trajectory, but it doesn't directly optimize the success rate of completing a task.

Thus, we decided to explore if the validation can be used to early stop training in order to obtain the highest success rate. If we show that validation loss cannot be used to find the highest success rate, then it might be useless to compute it, and it may even hurt performance, since training is done on a smaller subset of the original dataset.


<div style="display: flex; flex-direction: row;">
    <img src="https://cdn-uploads.huggingface.co/production/uploads/66583df28724def2ab9d231d/0kqagQNFSeaDBSPN5T73n.gif" alt="Transfer Cube" style="width: 45%; margin-right: 5px;">
    <img src="https://cdn-uploads.huggingface.co/production/uploads/66583df28724def2ab9d231d/hRjVdGvBb8qiZpKtvSkiE.gif" alt="PushT" style="width: 45%;">
</div>

Our experiments were conducted in two commonly used simulation environments: PushT and Aloha Transfer Cube, with two different policies: Diffusion and ACT (Action chunking with transformers). Simulation allows us to accurately compute the success rate at every 10K checkpoints, which is challenging in real environments.

## PushT

### Experimental Setup

The diffusion policy was trained on the PushT dataset, with 206 episodes at 10 FPS, yielding a total of 25,650 frames with an average episode duration of 12 seconds.

We used the same hyperparameters as the authors of Diffusion Policy. We trained the policy with three different seeds. 

Training for 100K steps plus evaluation every 10K steps took about 5 hours on two NVIDIA H100 80GB HBM3. Running evaluation and calculating success rates is the most costly part, taking on average 15 minutes at each batch rollout.

### Results

With PushT, we notice a divergent pattern for validation loss compared to the success rate :

Initially, validation loss increases after the first 10,000 steps and does not recover to its initial level by 60,000 steps. In contrast, despite the increase in validation loss, the success rates consistently improve between 10,000 and 60,000 steps across all seed runs.

During training, evaluation is done in simulated environments every 10K steps. We roll out the policy for 50 episodes and calculate the success rate.
<div style="display: flex; justify-content: space-between;">
    <div style="text-align: center; width: 32%;">
        <img src="https://cdn-uploads.huggingface.co/production/uploads/66583df28724def2ab9d231d/nPLhYwvsHSzPYW5959Tf0.png" alt="PushT Success Rate" style="width: 100%;">
        <p style="font-size: 12px;">PushT Success Rate</p>
    </div>
    <div style="text-align: center; width: 32%;">
        <img src="https://cdn-uploads.huggingface.co/production/uploads/66583df28724def2ab9d231d/AKvg0siHa8OlceKuj9A64.png" alt="PushT Validation Loss" style="width: 100%;">
        <p style="font-size: 12px;">PushT Validation Loss</p>
    </div>
    <div style="text-align: center; width: 32%;">
        <img src="https://cdn-uploads.huggingface.co/production/uploads/66583df28724def2ab9d231d/sakN2H_Mo7P4K4r3lds9Z.png" alt="PushT Training Loss" style="width: 100%;">
        <p style="font-size: 12px;">PushT Training Loss</p>
    </div>
</div>
To confirm that there's no correlation between validation loss and success rate, we run costly evaluations on 500 episodes to have more samples and decrease variance. We evaluate the checkpoints at 20K steps, at 50K steps, and at 90K steps. These are our results : 

| Step                   | 20K steps | 50K steps | 90K steps |
|------------------------|-----------|-----------|-----------|
| Mean Success Rate (%)  | 40.47     | 62.8      | 50.86     |
| Success Rate Standard Deviation      | 1.63      | 2.74      | 7.18      |
| Mean Validation Loss   | 0.0412    | 0.0965    | 0.56      |
| Validation Loss Standard Deviation   | 0.0013    | 0.0058    | 0.0008    |

The validation losses are more than twice as high at 50K steps compared to after 20K training steps, while the success rates improve by over 50% on average. Furthermore, the validation loss decreases between 50K and 90K steps, but the success rate decreases as well.

This correlation is the opposite of what is usually observed when a low validation loss indicates high performance.

## Transfer Cube

### Experimental Setup

In the second simulation, we used the Aloha arms environment on the Transfer-Cube task, with 50 episodes of human-recorded data.

Each episode consisted of 400 frames at 50 FPS, resulting in 8-second episodes captured with a single top-mounted camera.

We used the same hyperparameters as the authors of ACT. Same as for PushT, we trained the policy with three different seeds. 

Training for 100K steps + evaluation every 10K steps took about 6 hours on two NVIDIA H100 80GB HBM3. Running evaluation and calculating success rates is still the most costly part, in this task taking on average 20 minutes at each batch rollout.

### Results

In the case of Transfer Cube, we notice that while the validation loss plateaus, the success rate continues to grow.
<p style="font-size: 15px;">During training, evaluation is deployed in simulated environments every 10K steps. We roll out the policy for 50 episodes and calculate the success rate.</p> 
<div style="display: flex; justify-content: space-between;">
    <div style="text-align: center; width: 32%;">
        <img src="https://cdn-uploads.huggingface.co/production/uploads/66583df28724def2ab9d231d/J9ovFk63mYlmSIAOqYfUB.png" alt="Transfer Cube Success Rate" style="width: 100%;">
        <p style="font-size: 12px;">Transfer Cube Success Rate</p>
    </div>
    <div style="text-align: center; width: 32%;">
        <img src="https://cdn-uploads.huggingface.co/production/uploads/66583df28724def2ab9d231d/sGCApigxgmqhUoyKtPAHA.png" alt="Transfer Cube Validation Loss" style="width: 100%;">
        <p style="font-size: 12px;">Transfer Cube Validation Loss</p>
    </div>
    <div style="text-align: center; width: 32%;">
        <img src="https://cdn-uploads.huggingface.co/production/uploads/66583df28724def2ab9d231d/5u5MEtvuiHWP4Kf4Hg6Ii.png" alt="Transfer Cube Training Loss" style="width: 100%;">
        <p style="font-size: 12px;">Transfer Cube Training Loss</p>
    </div>
</div>
The success rate computed during training is highly variant (average of only 50 evaluation episodes), which is why we run additional evaluations on 500 episodes. 

For all three seeds, we calculate the success rate on the checkpoint at 30K steps and at 100K steps.

| Step                   | 30K steps | 100K steps |
|------------------------|-----------|------------|
| Mean Success Rate (%)  | 67.66     | 76.47      |
| Success Rate Standard Deviation    | 1.85      | 2.87       |
| Mean Validation Loss   | 0.2289    | 0.2243     |
| Validation Loss Standard Deviation   | 0.0112    | 0.0079     |

So while the validation loss stays roughly the same, or decreases by 2%, the success rate increases by about 10%. It is challenging to early-stop based on such fine-grained information; for our task it doesn't appear to be effective.

## Conclusion

Our experiments reveal a significant discrepancy between validation loss and task success rate metrics. On our tasks, we should not use validation loss to early stop training. This strategy does not ensure the highest success rate. The only alternative is to evaluate a maximum number of checkpoints under a compute/time budget. Of course, this is especially challenging in real-world cases, where evaluating a checkpoint on a real robot is expensive, and variance is high.

In the real world, there's no way to measure how well a policy performs. Instead, it's best to focus on quantitative progress and interpret signals like the fluidity of the robot's movements and the features that the policy learns.

For instance, when training *PollenRobotics' Reachy2* [https://x.com/RemiCadene/status/1798474252146139595] to transfer a cup from a rack to a person sitting on the opposite side, we noticed that the policy gradually learned more advanced concepts and trajectories:

- At the beginning, the robot was only able to grasp the object.
- Then it learned to rotate to give the cup to the person.
- Finally, it learned to rotate into the desired position and complete the full trajectory.

We see that progress is made of qualitative enhancements, and we can't completely translate these into qualitative directives.

This shows the need for a stable policy that will constantly improve with the number of training steps.

Developing another training framework, which would leverage implementation efficiency (easily computable metrics) while providing an accurate measure of how capable a policy is, is another improvement that could be investigated.

A notable effort by Li et al. [Evaluating Real-World Robot Manipulation Policies: arxiv.org/pdf/2405.05941] involves creating a highly realistic digital twin and evaluating real-world policies on it. They create highly realistic simulation environments with optimal physics and texture matching.

They achieve very good results transferring the learning from simulation to real-world, and the success rates in those simulated environments highly correlate with the success rates of real-world environments.

But having a good simulator is challenging and requires a thorough implementation for each new task and agent. As we learn to generate digital twins more easily, scaling this approach could lead to more efficient training and reduce time/resource costs of evaluation in the real world.

## References

* [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://www.ri.cmu.edu/pub_files/2011/4/Ross-AISTATS11-NoRegret.pdf)
* [Robomimic](https://ai.stanford.edu/blog/robomimic/)
* [Evaluating Real-World Robot Manipulation Policies in Simulation](https://arxiv.org/pdf/2108.03298)
* [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/pdf/2304.13705)
* [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/pdf/2303.04137)