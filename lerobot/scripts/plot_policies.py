import json
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from datetime import datetime

def plot_pc_success(json_files):
    pc_success_values = []
    labels = []

    for idx, file_path in enumerate(json_files):
        with open(file_path, 'r') as file:
            data = json.load(file)
            aggregated_data = data.get('aggregated', {})
            pc_success = aggregated_data.get('pc_success', 0)
            pc_success_values.append(pc_success)
            path_obj = Path(file_path)
            parent_dire_name = path_obj.parent.name
            labels.append(parent_dire_name)

    colors = plt.cm.viridis(np.linspace(0, 1, len(json_files)))
    bars = plt.bar(labels, pc_success_values, color=colors)
    plt.bar(labels, pc_success_values, color=colors)
    plt.xlabel('Policy')
    plt.ylabel('PC Success')
    plt.title('PC Success across policies')
    plt.xticks(fontsize=6)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2), ha='center', va='bottom')

    plt.tight_layout()

    # Generate the directory path and filename based on the current date and time
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    directory_path = 'outputs/plots'
    filename = f'pc_success_{date_time_str}.png'
    full_path = os.path.join(directory_path, filename)

    # Create the directory if it doesn't exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Save the plot
    plt.savefig(full_path)
    plt.close()  # Close the plot to free memory


if __name__ == '__main__':
    json_files = ['outputs/eval/2024-06-21/layer_wise_sum_transfer_cube/eval_info.json', 
                  'outputs/eval/2024-06-19/film_updated_eval/eval_info.json', 
                  'outputs/eval/2024-06-19/noembedding_act_transfer_cube_joint_dataset/eval_info.json']
    plot_pc_success(json_files)