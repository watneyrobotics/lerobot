import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Apply a beautiful Seaborn theme
sns.set_theme(style="darkgrid")

def smooth_data(data, window_size=5):
    # Apply a rolling mean to smooth the data
    smoothed_data = data.rolling(window=window_size, min_periods=1).mean()
    return smoothed_data

def plot_success_rate(csv_file, base_model_name, output_dir='outputs/plots'):
    # Load the CSV data
    data = pd.read_csv(csv_file)

    # Columns related to the specified model name
    columns = [
        f'{base_model_name}_84 - eval/pc_success',
        f'{base_model_name}_85 - eval/pc_success',
        f'{base_model_name}_1000 - eval/pc_success'
    ]
    
    data['mean_success'] = data[columns].mean(axis=1)
    data['min_success'] = data[columns].min(axis=1)
    data['max_success'] = data[columns].max(axis=1)

    # Create a Seaborn plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=data['Step'], y=data['mean_success'], label='Mean Success Rate')
    plt.fill_between(data['Step'], data['min_success'], data['max_success'], alpha=0.3, label='Range')

    # Add labels and title
    plt.xlabel('Step')
    plt.ylabel('Success Rate')
    plt.title(f'Success Rate Across Steps')
    plt.legend()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_file = f'{output_dir}/{base_model_name}_success_rate.png'
    plt.savefig(output_file)

    print(f'Plot saved to: {output_file}')

    # Show the plot
    plt.show()

def plot_validation_loss(csv_file, base_model_name, output_dir='outputs/plots', smooth_window=5):
    # Load the CSV data
    data = pd.read_csv(csv_file)

    # Columns related to the specified model name for validation loss
    columns = [
        f'{base_model_name}_1000 - val/loss',
        f'{base_model_name}_85 - val/loss',
        f'{base_model_name}_84 - val/loss'
    ]
    
    # Smooth all data across the specified columns
    smoothed_data = data[columns].rolling(window=smooth_window, min_periods=1).mean()

    # Smooth min and max values across the columns
    smoothed_min_val = data[columns].min(axis=1).rolling(window=smooth_window, min_periods=1).mean()
    smoothed_max_val = data[columns].max(axis=1).rolling(window=smooth_window, min_periods=1).mean()
    data['mean_val_loss'] = smoothed_data.mean(axis=1)

    # Create a Seaborn plot
    plt.figure(figsize=(10, 6))

    # Plot each smoothed series
    sns.lineplot(x=data['Step'], y=data['mean_val_loss'], label=f'Mean Validation Loss')

    # Plot smoothed min and max values in background
    plt.fill_between(data['Step'], smoothed_min_val, smoothed_max_val, alpha=0.3, label='Range')

    # Add labels and title
    plt.xlabel('Step')
    plt.ylabel('Validation Loss')
    plt.title(f'Validation Loss Across Steps')
    plt.legend()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_file = f'{output_dir}/{base_model_name}_validation_loss.png'
    plt.savefig(output_file)

    # Show the plot
    plt.show()

def plot_train_loss(csv_file, base_model_name, output_dir='outputs/plots', smooth_window=5):
    # Load the CSV data
    data = pd.read_csv(csv_file)
    data = data.iloc[30:]

    # Columns related to the specified model name for validation loss
    columns = [
        f'{base_model_name}_1000 - train/loss',
        f'{base_model_name}_85 - train/loss',
        f'{base_model_name}_84 - train/loss'
    ]
    
    # Smooth all data across the specified columns
    smoothed_data = data[columns].rolling(window=smooth_window, min_periods=1).mean()
    data['mean_train_loss'] = smoothed_data.mean(axis=1)
    # Smooth min and max values across the columns
    smoothed_min_train = data[columns].min(axis=1).rolling(window=smooth_window, min_periods=1).mean()
    smoothed_max_train = data[columns].max(axis=1).rolling(window=smooth_window, min_periods=1).mean()


    # Create a Seaborn plot
    plt.figure(figsize=(10, 6))

    # Plot mean training loss on a logarithmic scale
    sns.lineplot(x=data['Step'], y=data['mean_train_loss'], label=f'Mean Training Loss')

    # Plot smoothed min and max values in background on a logarithmic scale
    plt.fill_between(data['Step'], smoothed_min_train, smoothed_max_train, alpha=0.3, label='Range')

    # Add labels and title
    plt.xlabel('Step')
    plt.ylabel('Training Loss (log scale)')
    plt.title(f'Training Loss Across Steps')
    plt.legend()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_file = f'{output_dir}/{base_model_name}_train_loss_without_outliers.png'
    plt.savefig(output_file)

    # Show the plot
    plt.show()

# Example usage
csv_file = '/Users/mbar/Desktop/projects/huggingface/experiments/csv/wandb_export_2024-07-01T14_03_14.709+02_00.csv'
output_dir = '/Users/mbar/Desktop/projects/huggingface/experiments/plots'
base_model_name = 'compare_val_loss_transfer_cube'
#plot_validation_loss(csv_file, base_model_name, output_dir=output_dir)

#plot_success_rate(csv_file, base_model_name, output_dir=output_dir)

plot_train_loss(csv_file, base_model_name, output_dir=output_dir)
