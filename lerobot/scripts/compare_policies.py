import json
import numpy as np
import os
import datetime

def compute_statistics(values):
    """
    Computes the mean, variance, and standard deviation of 'pc_success' values across multiple JSON files.

    Parameters:
    - json_files: List of paths to JSON files.

    Returns:
    - A dictionary with 'mean', 'variance', and 'std_deviation' of 'pc_success' values.
    """
    pc_success_values = []

    for value in values:
        if isinstance(value, str) and "/" not in value:
            pc_success_values.append(float(value))
        elif isinstance(value, (int, float)):
            pc_success_values.append(value)
        else:
            with open(value, 'r') as file:
                data = json.load(file)
                aggregated_data = data.get('aggregated', {})
                if 'pc_success' in aggregated_data:
                    pc_success_values.append(aggregated_data['pc_success'])

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
    input_data = {"insertion_no_token":
                  [
                    '/admin/home/marina_barannikov/projects/lerobot/outputs/eval/2024-06-25/insertion_aloha_multidataset_no_token_16_seed/eval_info.json', 
                    '/admin/home/marina_barannikov/projects/lerobot/outputs/eval/2024-06-25/insertion_aloha_multidataset_no_token_17_seed/eval_info.json', 
                    '24.4'],
                "transfer_cube_no_token":
                [83.2, 74.0, 68.8],
                "base_insertion":
                [13.200000000000001, 18.0, 18.0],
                }
    statistics_results = {}

    # Compute statistics for each list in the input dictionary
    for key, values in input_data.items():
        # Convert numerical strings to float if necessary
        processed_values = []
        for value in values:
            try:
                processed_values.append(float(value))
            except ValueError:
                processed_values.append(value)

        statistics_results[key] = compute_statistics(processed_values)

    # Create output directory if it doesn't exist
    output_dir = '/admin/home/marina_barannikov/projects/lerobot/outputs/stats'
    os.makedirs(output_dir, exist_ok=True)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    # Save statistics to a JSON file
    output_path = os.path.join(output_dir, f'stats_{current_time}.json')
    with open(output_path, 'w') as outfile:
        json.dump(statistics_results, outfile, indent=4)

    print(f'Statistics saved to {output_path}')