import json
import json
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def compute_statistics(json_files):
    """
    Computes the mean, variance, and standard deviation of 'pc_success' values across multiple JSON files.

    Parameters:
    - json_files: List of paths to JSON files.

    Returns:
    - A dictionary with 'mean', 'variance', and 'std_deviation' of 'pc_success' values.
    """
    pc_success_values = []

    # Extract 'pc_success' values from each JSON file
    for file_path in json_files:
        with open(file_path, 'r') as file:
            data = json.load(file)
            aggregated_data = data.get('aggregated', {})
            if 'pc_success' in aggregated_data:
                pc_success_values.append(data['pc_success'])

    # Compute statistics
    mean = np.mean(pc_success_values)
    variance = np.var(pc_success_values)
    std_deviation = np.std(pc_success_values)

    return {
        'mean': mean,
        'variance': variance,
        'std_deviation': std_deviation
    }

# Example usage
json_files = ['file1.json', 'file2.json', 'file3.json']
statistics = compute_statistics(json_files)
print(statistics)

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

# Example usage
json_files = ['file1.json', 'file2.json', 'file3.json']
plot_pc_success(json_files)