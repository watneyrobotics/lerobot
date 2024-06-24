import json
import numpy as np
import matplotlib.pyplot as plt
import os
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
            labels.append(f'File {idx + 1}')

    colors = plt.cm.viridis(np.linspace(0, 1, len(json_files)))
    plt.bar(labels, pc_success_values, color=colors)
    plt.xlabel('JSON Files')
    plt.ylabel('PC Success')
    plt.title('PC Success across JSON Files')
    plt.xticks(rotation=45)
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
    json_files = ['file1.json', 'file2.json', 'file3.json']
    plot_pc_success(json_files)