import json
import torch

def get_dataset(input_array, json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    task_number = data.get('task')
    if task_number is None:
        raise ValueError('Invalid JSON file. Missing "task" key or value.')
    
    if not isinstance(task_number, int):
        raise ValueError('Invalid JSON file. "task" value should be an integer.')
    
    if task_number < 0 or task_number >= len(input_array):
        raise ValueError('Invalid task number. It should be within the range of the input array.')
    
    return input_array[task_number]
