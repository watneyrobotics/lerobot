import argparse
import datetime
import json
import subprocess

CPU_PER_TASK = 32
GPUS_PER_NODE = 1
ENV = "lerobot"  # name of the conda environment
TIME_PER_TASK = "03:59:00"
REPO_NAME = "git@github.com:marinabar/lerobot.git"

# Parameters
checkpoints = [
    "/fsx/marina_barannikov/outputs/train/compare_val_loss/pusht_84/080000",
]

output_dirs = [
    "/admin/home/marina_barannikov/projects/lerobot/outputs/eval/compare_val_loss/puhst_84_80k",
]

eval_args = [
    "eval.n_episodes=400 eval.use_async_envs=true eval.batch_size=100",
]

# Commit hashes to checkout the right code version
commits = [
    "main",
]

# Wwill be used for naming the sbatch jobs, logging
job_names = [
    "eval_pusht_84_80k",
]


# Get the current time and date for logging
current_time = datetime.datetime.now()
time = current_time.strftime("%Y-%m-%d-%H:%M:%S")


# Use this function to run the jobs from the lists defined above
def create_eval_job_from_dict(checkpoints, output_dirs, eval_args, commits, job_names):
    assert (
        len(checkpoints) == len(output_dirs) == len(eval_args) == len(commits) == len(job_names)
    ), f"All lists must have the number of elements, but have {len(checkpoints)}, {len(output_dirs)}, {len(eval_args)}, {len(commits)}, {len(job_names)}."

    job_parameters = {}
    for i in range(len(checkpoints)):
        checkpoint_path = checkpoints[i]
        output_path = output_dirs[i]
        eval_arg = eval_args[i]
        commit = commits[i]
        job_name = job_names[i]

        # Submit a job for each set of parameters
        job_parameters = run_eval_job(
            checkpoint_path, output_path, eval_arg, commit, job_name, job_parameters=job_parameters
        )

    # Save all sbatch commands to a JSON file
    json_filename = f"param_jobs/job_parameters_{time}.json"
    with open(json_filename, "w") as json_file:
        json.dump(job_parameters, json_file, indent=4)

    print(f"Saved sbatch commands to {json_filename}")


# Use this function to re-run the jobs from the JSON file that was saved
def json_to_dict(json_filename):
    with open(json_filename) as json_file:
        saved_jobs = json.load(json_file)

    checkpoints = [saved_jobs[job_name]["checkpoint"] for job_name in saved_jobs]
    output_dirs = [saved_jobs[job_name]["output_dir"] for job_name in saved_jobs]
    eval_args = [saved_jobs[job_name]["eval_arg"] for job_name in saved_jobs]
    commits = [saved_jobs[job_name]["commit"] for job_name in saved_jobs]
    job_names = list(saved_jobs.keys())

    create_eval_job_from_dict(checkpoints, output_dirs, eval_args, commits, job_names)


# This is the main function that submits a single job to the cluster
def run_eval_job(
    checkpoint_path, output_path, eval_arg, commit, job_name, custom_command=None, job_parameters=None
):
    if custom_command:
        command = custom_command
        commit = "main"
        job_name = "eval_" + time + "_" + "_".join(custom_command.split(" ")[3].split("/"))
        if "--out-dir" not in custom_command:
            custom_command += f" --out-dir lerobot/outputs/eval/{time}"
            print(
                "WARNING: The default output directory was in the temporary execution folder, new directory was created at lerobot/outputs/eval/{time}. To change the output directory, please provide the --out-dir argument."
            )
    else:
        if not all([checkpoint_path, output_path, eval_arg, commit, job_name]):
            raise ValueError("Missing arguments for command execution.")
        command = f"python lerobot/scripts/eval.py {eval_arg} -p {checkpoint_path} --out-dir {output_path}"

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
        "--container-image=/fsx/$USER/docker_images/huggingface+lerobot-gpu+dev.sqsh",
        "--container-mounts=/fsx/$USER",
        "--container-workdir=/admin/home/$USER/projects/lerobot",
        # Warning: the log output folder should be created before running the job !
        f"--output=/admin/home/marina_barannikov/projects/lerobot/outputs/eval/logs/{time}-{job_name}-%j.out",
        "--ntasks=1",
        # The command for launching the script creates a temporary directory, clones the repo,
        # checks out the right commit, runs the eval.py , and deletes the temporary folder up
        f'--wrap=bash -c "source /admin/home/$USER/miniconda3/bin/activate {ENV} && export GIT_LFS_SKIP_SMUDGE=1 && \
            mkdir -p /admin/home/$USER/lerobot_temp_{job_name} && \
            cd /admin/home/$USER/lerobot_temp_{job_name}  && \
            git clone {REPO_NAME} && \
            cd lerobot && \
            git reset --hard {commit} && \
            {command} && \
            cd ../.. && \
            rm -rf lerobot_temp_{job_name} && exit"',
    ]

    job_parameters[job_name] = {
        "checkpoint": checkpoint_path,
        "output_dir": output_path,
        "eval_arg": eval_arg,
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
        print(f"Running command: {custom_command}")
        run_eval_job(None, None, None, None, None, custom_command=custom_command)
    else:
        print("Running jobs from the parameters lists defined in the Python file.")
        create_eval_job_from_dict(checkpoints, output_dirs, eval_args, commits, job_names)
