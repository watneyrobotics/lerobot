import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Apply a beautiful Seaborn theme
sns.set_theme(style="darkgrid")


def smooth_data(data, window_size=5):
    # Apply a rolling mean to smooth the data
    smoothed_data = data.rolling(window=window_size, min_periods=1).mean()
    return smoothed_data

def plot_metrics(csv_file, base_model_name, plot_type, output_dir="outputs/plots", smooth_window=5):
    # Load the CSV data
    data = pd.read_csv(csv_file)

    # Define column names and labels based on plot_type
    if plot_type == "success_rate":
        columns = [
            f"{base_model_name}_84 - eval/pc_success",
            f"{base_model_name}_85 - eval/pc_success",
            f"{base_model_name}_1000 - eval/pc_success",
        ]
        y_label = "Success Rate"
        title = "Success Rate Across Steps"
        mean_col = "mean_success"
        std_col = "std_success"

    elif plot_type == "validation_loss":
        columns = [
            f"{base_model_name}_1000 - val/loss",
            f"{base_model_name}_85 - val/loss",
            f"{base_model_name}_84 - val/loss",
        ]
        y_label = "Validation Loss"
        title = "Validation Loss Across Steps"
        mean_col = "mean_val_loss"
        std_col = "std_val_loss"

    elif plot_type == "train_loss":
        # Remove outliers for train_loss plot
        data = data.iloc[30:]
        columns = [
            f"{base_model_name}_1000 - train/loss",
            f"{base_model_name}_85 - train/loss",
            f"{base_model_name}_84 - train/loss",
        ]
        y_label = "Training Loss"
        title = "Training Loss Across Steps"
        mean_col = "mean_train_loss"
        std_col = "std_train_loss"

    else:
        raise ValueError("Invalid plot_type. Choose from 'success_rate', 'validation_loss', or 'train_loss'.")

    # Smooth data and compute mean and standard deviation
    smoothed_data = data[columns].rolling(window=smooth_window, min_periods=1).mean()
    data[mean_col] = smoothed_data.mean(axis=1)
    data[std_col] = smoothed_data.std(axis=1)

    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=data["Step"], y=data[mean_col], label=f"Mean {y_label}")
    plt.fill_between(data["Step"], 
                     data[mean_col] - data[std_col], 
                     data[mean_col] + data[std_col], 
                     alpha=0.3, 
                     label="Standard Deviation")
    
    # Add labels and title
    plt.xlabel("Step")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot
    output_file = f"{output_dir}/{base_model_name}_{plot_type}.png"
    plt.savefig(output_file)
    plt.show()

    print(f"Plot saved to: {output_file}")


def plot_mse_loss(csv_file, base_model_name, output_dir="outputs/plots"):
    # Load the CSV data
    data = pd.read_csv(csv_file)
    
    # Group by 'step' and calculate mean and std for 'mse_loss'
    grouped_data = data.groupby("step")["mse_loss"].agg(["mean", "std"]).reset_index()

    # Create a Seaborn plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=grouped_data["step"], y=grouped_data["mean"], label="Mean MSE Loss")
    plt.fill_between(grouped_data["step"], 
                     grouped_data["mean"] - grouped_data["std"], 
                     grouped_data["mean"] + grouped_data["std"], 
                     alpha=0.3, 
                     label="Standard Deviation")

    # Add labels and title
    plt.xlabel("Step")
    plt.ylabel("MSE Loss")
    plt.title("MSE Loss Across Steps")
    plt.legend()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot
    output_file = f"{output_dir}/{base_model_name}_mse_loss.png"
    plt.savefig(output_file)

    print(f"Plot saved to: {output_file}")

    plt.show()


def plot_l1_loss(csv_file, base_model_name, output_dir="outputs/plots"):
    # Load the CSV data
    data = pd.read_csv(csv_file)

    # Group by 'step' and calculate mean, min, and max for 'mse_loss'
    grouped_data = data.groupby("step")["l1_loss"].agg(["mean", "min", "max"]).reset_index()

    # Create a Seaborn plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=grouped_data["step"], y=grouped_data["mean"], label="Mean L1 Loss")
    plt.fill_between(grouped_data["step"], grouped_data["min"], grouped_data["max"], alpha=0.3, label="Range")

    # Add labels and title
    plt.xlabel("Step")
    plt.ylabel("L1 Loss")
    plt.title("L1 Loss Across Steps")
    plt.legend()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot
    output_file = f"{output_dir}/{base_model_name}_l1_loss.png"
    plt.savefig(output_file)

    print(f"Plot saved to: {output_file}")

    # Show the plot
    plt.show()


# Example usage
csv_file = (
    "/Users/mbar/Desktop/projects/huggingface/experiments/csv/wandb_export_2024-07-01T12_46_39.804+02_00.csv"
)
output_dir = "/Users/mbar/Desktop/projects/huggingface/experiments/plots"
base_model_name = "compare_val_loss_transfer_cube"

plot_metrics(csv_file, base_model_name, "success_rate", output_dir=output_dir)

csv = "dev/pusht_results.csv"
#plot_mse_loss(csv, "pusht")
