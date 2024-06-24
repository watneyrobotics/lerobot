import json
import numpy as np

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


if __name__ == '__main__':
    json_files = ['file1.json', 'file2.json', 'file3.json']
    statistics = compute_statistics(json_files)