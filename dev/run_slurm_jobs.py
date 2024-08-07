"""
This is a script for submitting evaluation or training jobs to the cluster.
It can be used to send multiple jobs at once, with different parameters, and from different commits.
The script will create a temporary directory, clone the repository, checkout the right commit, run the script, and delete the temporary directory.

To start, define the name of the conda environment to activate and the repository name to clone.
The log directory for SLURM, defined in the sbatch command with --output, should be created before running the script.

The script can be run in three ways:

1. Construct the jobs from the lists defined in the script.

To do this, simply run the script without any arguments, after having defined parameters like the output directories, and config overrides.
This will call create_eval_job_from_dict() function, which will submit the jobs to the cluster.
To run the training jobs, call create_train_job_from_dict() function instead.
To use create_train_job_from_dict(), you need to define the overrides for the training script.

For example, add to job_args the following entry :
    hydra.job.name=test \
    hydra.run.dir=test \
    policy=act \
    dataset_repo_id=lerobot/aloha_sim_transfer_cube_human \
    env=aloha env.task=AlohaTransferCube-v0 \
    training.offline_steps=1000

2. Run a custom command.

Simply provide the command you would like to run as an argument to the script :
python dev/run_slurm_jobs.py \
    python lerobot/scripts/eval.py \
    eval.n_episodes=1 \
    eval.use_async_envs=false \
    eval.batch_size=1 \
    -p lerobot/diffusion_pusht \
    --out-dir /admin/home/marina_barannikov/projects/lerobot/outputs/eval/test/pusht


In this case, if the commit hash is not provided, the script will default to the main branch.

3. Re-run the jobs from a JSON file.
If you have submitted jobs through lists of arguments, and they failed, you can re-run them by providing the JSON file that was automatically saved.
In that case, you need to call the json_to_dict() function, and provide the path to the JSON file and whether it is an evaluation or training job.
"""

import argparse
import datetime
import json
import os
import subprocess

CPU_PER_TASK = 12
GPUS_PER_NODE = 1
ENV = "lerobot"  # name of the conda environment
TIME_PER_TASK = "05:59:00"
REPO_NAME = "git@github.com:marinabar/lerobot.git"
USER = os.environ["USER"]
WORKDIR = os.getcwd()

# Specific arguments for eval.py script
checkpoints = [
    "/fsx/marina_barannikov/outputs/train/compare_val_loss/pusht_84/080000",
]

output_dirs = [
    "/admin/home/marina_barannikov/projects/lerobot/outputs/eval/test",
]

# Common arguments for both eval.py and train.pu scripts
job_args = [
    "hydra.job.name=aloha_embed_token \
    hydra.run.dir=/fsx/marina_barannikov/outputs/multitask_08/aloha_embed_token \
    policy=act_multitask \
    env=aloha \
    policy.use_vae=true \
    wandb.enable=true",
]

# Commit hashes to checkout the right code version
commits = [
    "b6336065a550858225fd7f490ab97e87cf6551e8",
]

# Will be used for naming the sbatch jobs, logging
job_names = [
    "aloha_embed_token",
]


# Get the current time and date for logging
current_time = datetime.datetime.now()
time = current_time.strftime("%Y-%m-%d_%H-%M-%S")


# Use this function to run the jobs from the lists defined above
def create_eval_job_from_dict(checkpoints, output_dirs, eval_args, commits, job_names):
    assert (
        len(checkpoints) == len(output_dirs) == len(eval_args) == len(commits) == len(job_names)
    ), f"All lists must have the number of elements, but got {len(checkpoints)}, {len(output_dirs)}, {len(eval_args)}, {len(commits)}, {len(job_names)}."

    job_parameters = {}
    for i in range(len(checkpoints)):
        # Submit a job for each set of parameters
        job_parameters = run_job(
            script="eval",
            script_arg=eval_args[i],
            commit=commits[i],
            job_name=job_names[i],
            checkpoint_path=checkpoints[i],
            output_path=output_dirs[i],
            job_parameters=job_parameters,
        )

    # Save all sbatch commands to a JSON file
    json_filename = f"outputs/job_params/eval_slurm_jobs_{time}.json"
    os.makedirs(os.path.dirname(json_filename), exist_ok=True)

    with open(json_filename, "w") as json_file:
        json.dump(job_parameters, json_file, indent=4)

    print(f"Saved sbatch commands to {json_filename}")


def create_train_job_from_dict(train_args, commits, job_names):
    assert (
        len(train_args) == len(commits) == len(job_names)
    ), f"All lists must have the number of elements, but got {len(train_args)}, {len(commits)}, {len(job_names)}."

    job_parameters = {}
    for i in range(len(train_args)):
        # Submit a job for each set of parameters
        job_parameters = run_job(
            script="train",
            script_arg=train_args[i],
            commit=commits[i],
            job_name=job_names[i],
            job_parameters=job_parameters,
        )

    # Save all sbatch commands to a JSON file
    json_filename = f"outputs/param_jobs/train_slurm_jobs_{time}.json"
    os.makedirs(os.path.dirname(json_filename), exist_ok=True)

    with open(json_filename, "w") as json_file:
        json.dump(job_parameters, json_file, indent=4)

    print(f"Saved sbatch commands to {json_filename}")


# Use this function to re-run the jobs from the JSON file that was saved
def json_to_dict(json_filename, script):
    with open(json_filename) as json_file:
        saved_jobs = json.load(json_file)

    job_names = list(saved_jobs.keys())

    checkpoints = [saved_jobs[job_name].get("checkpoint") for job_name in job_names]
    output_dirs = [saved_jobs[job_name].get("output_dir") for job_name in job_names]
    job_args = [saved_jobs[job_name].get("job_arg") for job_name in job_names]
    commits = [saved_jobs[job_name].get("commit") for job_name in job_names]

    for i in range(len(job_names)):
        print(f"Re-running job {job_names[i]}")

        run_job(
            script=script,
            script_arg=job_args[i],
            commit=commits[i],
            job_name=job_names[i],
            checkpoint_path=checkpoints[i],
            output_path=output_dirs[i],
        )


# This is the main function that submits a single job to the cluster
def run_job(
    script,
    script_arg,
    commit,
    job_name,
    checkpoint_path=None,
    output_path=None,
    custom_command=None,
    job_parameters=None,
):
    if custom_command:
        commit = "main"
        command = custom_command
        if "eval.py" in custom_command:
            # Create job name based on the checkpoint path
            job_name = f"{time.split('_')[1]}_{command.split('-p')[1].split()[0].replace('/', '_')}"
            if "--out-dir" not in custom_command:
                out_dir = f"{WORKDIR}/outputs/eval/{time.split('_')[0]}/{job_name}"
                command += f" --out-dir {out_dir}"
                print(
                    "No output directory was specified, default directory was in the created temporary folder."
                )
                print(
                    f"New directory was created at {out_dir}. To change the output directory, provide the --out-dir argument."
                )
        else:
            # Create job name
            if "hydra.job.name" in custom_command:
                job_name = command.split("hydra.job.name=")[1].split(" ")[0]
            else:
                job_name = f"{time.split('_')[1]}_{command.split('dataset_repo_id=')[1].split()[0].replace('/', '_') if 'dataset_repo_id' in command else 'default'}"
            # Change default run directory
            if "hydra.run.dir" not in custom_command:
                command = command.replace(
                    "train.py",
                    f"train.py hydra.run.dir={WORKDIR}/outputs/train/{time.split('_')[0]}/{job_name}",
                )
                print(
                    "No run directory was specified, default directory was in the created temporary folder."
                )
                print(
                    f"New directory was created at {WORKDIR}/outputs/train/{time.split('_')[0]}/{job_name}. To change the run directory, provide the hydra.run.dir argument."
                )

        print(f"Running custom command: {command}")
    else:
        if script == "eval":
            if not all([checkpoint_path, output_path, script_arg, commit, job_name]):
                raise ValueError("Missing arguments for evaluation.")
            command = (
                f"python lerobot/scripts/eval.py {script_arg} -p {checkpoint_path} --out-dir {output_path}"
            )
        elif script == "train":
            if not all([script_arg, commit, job_name]):
                raise ValueError("Missing arguments for training.")
            command = f"python lerobot/scripts/train.py {script_arg}"
        else:
            raise ValueError("Invalid script name. Please provide either 'eval' or 'train'.")

    # Create dictionary for keeping track of the jobs
    if not job_parameters:
        job_parameters = {}

    # Construct the sbatch command
    sbatch_cmd = [
        "sbatch",
        "--qos=high",
        f"--job-name={job_name}",
        f"--cpus-per-task={CPU_PER_TASK}",
        f"--gres=gpu:{GPUS_PER_NODE}",
        f"--time={TIME_PER_TASK}",
        "--partition=hopper-prod",
        f"--container-image=/fsx/{USER}/docker_images/huggingface+lerobot-gpu+dev.sqsh",
        f"--container-mounts=/fsx/{USER}",
        f"--container-workdir=/admin/home/{USER}/projects/lerobot",
        # Warning: the log output folder should be created before running the job !
        f"--output=/admin/home/marina_barannikov/projects/lerobot/outputs/slurm_logs/{job_name}-%j.out",
        "--ntasks=1",
        # The command for launching the script creates a temporary directory, clones the repo,
        # checks out the right commit, runs the eval.py , and deletes the temporary folder up
        f'--wrap=bash -c "source /admin/home/{USER}/miniconda3/bin/activate {ENV} && export GIT_LFS_SKIP_SMUDGE=1 && \
            mkdir -p {WORKDIR}/../lerobot_temp_{job_name} && \
            cd {WORKDIR}/../lerobot_temp_{job_name}  && \
            git clone {REPO_NAME} && \
            cd lerobot && \
            git reset --hard {commit} && pip install . && \
            {command} && \
            cd ../.. && \
            rm -rf lerobot_temp_{job_name} && exit"',
    ]

    if script == "eval":
        job_parameters[job_name] = {
            "checkpoint": checkpoint_path,
            "output_dir": output_path,
            "job_arg": script_arg,
            "commit": commit,
            "job_name": job_name,
            "sbatch_cmd": " ".join(sbatch_cmd),  # Optional: Store the sbatch command string for reference
        }
    else:
        job_parameters[job_name] = {
            "run_dir": script,
            "job_arg": script_arg,
            "commit": commit,
            "job_name": job_name,
            "sbatch_cmd": " ".join(sbatch_cmd),  # Optional: Store the sbatch command string for reference
        }

    # Submit the job using subprocess
    result = subprocess.run(sbatch_cmd, capture_output=True, text=True)

    if result.returncode == 0:
        job_id = result.stdout.strip()
        print(f"{job_id} for task {job_name}")
    else:
        print(f"Failed to submit job for task {job_name}:")
        print(result.stderr)

    return job_parameters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation jobs on the cluster")
    parser.add_argument(
        "c",
        nargs=argparse.REMAINDER,
        default=None,
        help="Optional command to run. If not provided, the script will run the jobs from the lists above.",
    )

    args = parser.parse_args()
    if args.c:
        custom_command = " ".join(args.c)
        run_job(None, None, None, None, None, None, custom_command)
    else:
        print("Running jobs from the parameters lists defined in the Python file.")
        # create_eval_job_from_dict(checkpoints, output_dirs, job_args, commits, job_names)
        create_train_job_from_dict(job_args, commits, job_names)
