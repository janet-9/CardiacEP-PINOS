'''
Script for comparing the training, evaluation and compute time for each of the trained models in a folder 
'''
#Import required modules for constructing and training the model.
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
import argparse
import os
from pathlib import Path
import plotly
import pandas as pd
#import networkx
#imports required for data logging
import json
import re

# Define the optional arguments to use when calling the operator
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Arguments for the data structure:
    # Arguments for the FNO and training parameters 
    parser.add_argument('-b', '--batch-size', dest='batch_size', default = 15, help='Batch size for evaluating the data. Default is %(default)s')
    parser.add_argument('-m', '--modes', dest='modes', default = 16, help='Number of modes to keep during Fourier transform. Default is %(default)s')
    parser.add_argument('-hc', '--hidden-channels', dest='hidden_channels', default = 32, help='Number of hidden channels for the FNO. Default is %(default)s')
    parser.add_argument('-em', '--eval-metric', dest='eval_metric', default = 'mse', help='Evaluation metric that was used for saving the best model. Default is %(default)s')
    parser.add_argument('-ch', '--channels', dest = 'channels', type = int, default = 1, help = 'Number of channels to use in training. Default is %(default)s (Voltage only). Change to 2 to include W (recovery current).' )
    parser.add_argument('-phys', '--phys-loss', dest = 'phys_loss', type = float, default = 0.0, help = 'Weighting for the physics loss. Default is %(default)s')
    parser.add_argument('-p_meth', '--phys-meth', dest = 'phys_method', type = str, choices= {'finite_difference', 'finite_difference_stat', 'autograd'},  default = 'finite_difference_stat', help = 'Method for calculating the physics loss. Default is %(default)s')
    
    args = parser.parse_args()


## ======================================================================= ##
# == DEFINING THE FUNCTIONS  == ##
## ======================================================================= ##


## ----------------------------------------------------------------------- ##
# EXTRACTION OF THE DATA
## ----------------------------------------------------------------------- ##


## Extract all the available training logs ## 
def collect_training_logs(base_dir="."):
    """
    Collects all training_log.json files under folders starting with 'Results'.
    Returns a list of (folder_name, training_data_dict).
    """
    logs = []
    for folder in os.listdir(base_dir):
        if folder.startswith("Results") and os.path.isdir(os.path.join(base_dir, folder)):
            print(f"Results folder found: {folder}")
            log_path = os.path.join(base_dir, folder, "training_log.json")
            if os.path.exists(log_path):
                try:
                    with open(log_path, "r") as f:
                        data = json.load(f)
                        logs.append((folder, data))
                except Exception as e:
                    print(f"[Warning] Could not load {log_path}: {e}")
    return logs

## Extract the MSE results ## 
def collect_error_logs(base_dir="."):
    """
    Collects all mse_per_timestep.json files under folders starting with 'Results'.
    Returns a list of (folder_name, mse data).
    """
    logs = []
    for folder in os.listdir(base_dir):
        if folder.startswith("Results") and os.path.isdir(os.path.join(base_dir, folder)):
            log_path = os.path.join(base_dir, folder, "mse_per_timestep.json")
            if os.path.exists(log_path):
                try:
                    with open(log_path, "r") as f:
                        data = json.load(f)
                        logs.append((folder, data))
                except Exception as e:
                    print(f"[Warning] Could not load {log_path}: {e}")
    return logs

# Extract the physics loss weighting 
def extract_pino_number(folder_name):
    """
    Extract the numeric value from a folder name like 'PINO_0.1'.
    Returns None if pattern doesn't match.
    """
    #match = re.search(r'PINO_([\d\.]+)', folder_name)
    match = re.search(r'_([\d\.]+)_D', folder_name)
    if match:
        return match.group(1)
    return None


# Collect the recorded metrics from the best performing model during training
def collect_json_results(base_dir="."):
    logs = []
    for folder in os.listdir(base_dir):
        if folder.startswith("Results") and os.path.isdir(os.path.join(base_dir, folder)):
            log_path = os.path.join(base_dir, folder, "Best_Model_Results.json")
            if os.path.exists(log_path):
                try:
                    with open(log_path, "r") as f:
                        data = json.load(f)
                        logs.append((folder, data))
                except Exception as e:
                    print(f"[Warning] Could not load {log_path}: {e}")
    return logs

# Collect the data from the cell plots...
def collect_json_cells(base_dir="."):
    logs = []
    for folder in os.listdir(base_dir):
        if folder.startswith("Results") and os.path.isdir(os.path.join(base_dir, folder)):
            log_path = os.path.join(base_dir, folder, "cell_time_series.json")
            if os.path.exists(log_path):
                try:
                    with open(log_path, "r") as f:
                        data = json.load(f)
                        logs.append((folder, data))
                except Exception as e:
                    print(f"[Warning] Could not load {log_path}: {e}")
    return logs

## ----------------------------------------------------------------------- ##
# PLOTTING FUNCTIONS
## ----------------------------------------------------------------------- ##

def make_label(folder: str, eval_metric: str) -> str:
    f = folder.lower()
    if "physics" in f:
        base = "Physics"
    elif "data" in f:
        base = "Data"
    elif "fine" in f:
        base = "Fine_Tuned"
    elif "vloss" in f:
        p = extract_pino_number(folder) or "unknown"
        base = f"PINO_vloss{p}"
    elif "pino" in f:
        p = extract_pino_number(folder) or "unknown"
        base = f"PINO_{p}"
    else:
        base = folder

    suffix = "_combined" if "combined" in f else f"_{eval_metric}"
    return f"{base}{suffix}"

def plot_combined_logs_training(logs):
    """
    Plot training losses from multiple logs on a single figure.
    Each line is labeled using the folder name.
    """
    cmap = plt.colormaps["tab10"]
    linestyles = ['-', '--', '-.', ':']
    color_idx = 0

    plt.figure(figsize=(12, 7))
    
    for folder, data in logs:
               #label = folder
        color = cmap(color_idx % 10)
        linestyle = linestyles[color_idx % len(linestyles)]
        color_idx += 1
        linewidth = 1
        # Extract label from folder name
        label = make_label(folder, args.eval_metric)
        if "data" in folder.lower():
            color = 'black'
            linestyle = '-'

        epochs = [entry["epoch"] for entry in data]
        avg_loss = [entry["avg_loss"] for entry in data]
        # Plot losses - Training Losses
        plt.plot(epochs, avg_loss, label=f"{label} - Train", color=color, linestyle= linestyle, linewidth = linewidth)
        
    plt.title("Training Losses Across All Runs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("Model_Evalaution_Train.png")
    plt.show()


def plot_combined_logs_testing(logs):
    """
    Plot evaluation losses from multiple logs on a single figure.
    Each line is labeled using the folder name.
    """
    cmap = plt.colormaps["tab10"]
    linestyles = ['-', '--', '-.', ':']
    color_idx = 0

    plt.figure(figsize=(12, 7))
    
    for folder, data in logs:
        #label = folder
        color = cmap(color_idx % 10)
        linestyle = linestyles[color_idx % len(linestyles)]
        color_idx += 1
        linewidth = 1
        
        label = make_label(folder, args.eval_metric)
        if "data" in folder.lower():
            color = 'black'
            linestyle = '-'
        
        epochs = [entry["epoch"] for entry in data]
        

        # Load the testing errors throughout the training process
        h1 = [next((v for k, v in entry.items() if re.search(r"_h1$", k)), None) for entry in data]
        l2 = [next((v for k, v in entry.items() if re.search(r"_l2$", k)), None) for entry in data]
        mse = [next((v for k, v in entry.items() if re.search(r"_mse$", k)), None) for entry in data]
        ap = [next((v for k, v in entry.items() if re.search(r"_ap_phys$", k)), None) for entry in data]
        combined = [next((v for k, v in entry.items() if re.search(r"_combined$", k)), None) for entry in data]

        # Plot losses - Testing (mse for testing for all models)
        plt.plot(epochs, mse, label=f"{label} - Test - mse", color=color, linestyle=linestyle, linewidth = linewidth)
    
        if any(h1): plt.plot(epochs, h1, label=f"{label} - H1", color=color, linestyle="--")
        if any(l2): plt.plot(epochs, l2, label=f"{label} - L2", color=color, linestyle=":")
        if any(mse): plt.plot(epochs, mse, label=f"{label} - mse", color=color, linestyle="-.")
        if any(ap): plt.plot(epochs, ap, label=f"{label} - AP", color=color, linestyle="solid", alpha=0.5)
        if any(combined): plt.plot(epochs, combined, label=f"{label} - Combined", color=color, linestyle="dotted")
        
    plt.title("Testing Losses Across All Runs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("Model_Evaluation_Test.png")
    plt.show()


def plot_combined_errors(logs):
    """
    Plot training and evaluation losses from multiple logs on a single figure.
    Each line is labeled using the folder name.
    """
    cmap = plt.colormaps["tab10"]
    linestyles = ['-', '--', '-.', ':']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    channels = ["Voltage", "Recovery"]

    for color_idx, (folder, data) in enumerate(logs):
        linestyle = linestyles[color_idx % len(linestyles)]
        color = cmap(color_idx % 10)
        linewidth = 1
        label = make_label(folder, args.eval_metric)
        if "data" in folder.lower():
            color = 'black'
            linestyle = '-'
       
        mse = data["mse"]  # This is a dict with "Voltage" and "Recovery" keys
        
        for i, ch in enumerate(channels):
            ax = axes[i]
            timesteps = np.arange(len(mse[ch]))
            ax.plot(timesteps, mse[ch], color=color, label=label, linestyle = linestyle, linewidth=linewidth)
            ax.set_title(f"{ch} MSE Over Time")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Mean Squared Error")
            ax.grid(False)

    axes[0].legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig("Model_Evaluation_Errors.png", dpi=300)
    plt.show()

def plot_combined_errors_unseen(logs):
    """
    Plot training and evaluation losses from multiple logs on a single figure.
    Each line is labeled using the folder name.
    """
    cmap = plt.colormaps["tab10"]
    linestyles = ['-', '--', '-.', ':']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    channels = ["Voltage", "Recovery"]

    for color_idx, (folder, data) in enumerate(logs):
        linestyle = linestyles[color_idx % len(linestyles)]
        color = cmap(color_idx % 10)
        linewidth = 1
        label = make_label(folder, args.eval_metric)
        if "data" in folder.lower():
            color = 'black'
            linestyle = '-'
       
        timesteps = data["timesteps"]
        mse = data["mse"]  # This is a dict with "Voltage" and "Recovery" keys

        for i, ch in enumerate(channels):
            ax = axes[i]
            ax.plot(timesteps, mse[ch], color=color, label=label, linestyle = linestyle, linewidth=linewidth)
            ax.set_title(f"{ch} MSE Over Time")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Mean Squared Error")
            ax.set_xlim(200, len(timesteps))
            ax.grid(False)

    axes[0].legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig("Model_Evaluation_Errors_Unseen.png", dpi=300)
    plt.show()


def plot_results_avg(logs):
    """
    Plot grouped bar chart comparing Voltage error metrics across different models.
    """
    cmap = plt.colormaps["tab10"]
    channel_key = "Channel 0 (0 = Voltage, 1 = Recovery Current)"

    # Define which metrics to extract
    avg_error_metrics = [
        "Avg Mean Absolute Error",
        "Average Cell Error",
        #"Average Error of Top 10 Worst Cells",
        "Mean Signed Error (Bias)",
        #"Max Signed Error",
        #"Pct Errors > Avg MAE",
        #"Pct Cells with Avg Error > Avg Cell Error"
    ]

    folder_labels = []
    all_errors = {metric: [] for metric in avg_error_metrics}

    for color_idx, (folder, data) in enumerate(logs):
        label = make_label(folder, args.eval_metric)
        if "data" in folder.lower():
            color = 'black'
            linestyle = '-'

        folder_labels.append(label)

        ch_data = data[channel_key]
        for metric in avg_error_metrics:
            value = ch_data.get(metric, None)
            if value is not None:
                all_errors[metric].append(value)
            else:
                all_errors[metric].append(0.0)  # fallback if metric missing

    # --- Plotting ---
    
    group_spacing = 0.65             
    x = np.arange(len(folder_labels)) * group_spacing
    num_metrics = len(avg_error_metrics)

    total_width = group_spacing * 0.9  # 90% of that spacing used by bars
    bar_width  = total_width / num_metrics

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, metric in enumerate(avg_error_metrics):
        offset = (i - (num_metrics-1)/2) * bar_width
        ax.bar(x + offset, all_errors[metric], bar_width, label=metric)

    ax.set_xticks(x, folder_labels)
    ax.margins(x=0)
    ax.set_xlim(x[0] - total_width/2, x[-1] + total_width/2)
    ax.set_ylabel("Error Value")
    ax.set_title("Voltage Error Metrics Comparison")
    ax.set_xticklabels(folder_labels, ha='right')
    ax.legend(loc = "best")
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig("Model_Voltage_Error_Metrics_Comparison.png", dpi=300)
    plt.show()

def plot_results_pct(logs):
    """
    Plot grouped bar chart comparing Voltage error metrics across different models.
    """
    cmap = plt.colormaps["tab10"]
    channel_key = "Channel 0 (0 = Voltage, 1 = Recovery Current)"

    error_metrics = [
        "Pct Errors > Avg MAE",
        "Pct Cells with Avg Error > Avg Cell Error"
    ]

    folder_labels = []
    all_errors = {metric: [] for metric in error_metrics}

    for color_idx, (folder, data) in enumerate(logs):
        label = make_label(folder, args.eval_metric)
        if "data" in folder.lower():
            color = 'black'
            linestyle = '-'

        
        folder_labels.append(label)

        ch_data = data[channel_key]
        for metric in error_metrics:
            value = ch_data.get(metric, None)
            if value is not None:
                all_errors[metric].append(value)
            else:
                all_errors[metric].append(0.0)  # fallback if metric missing

    # --- Plotting ---
    group_spacing = 0.65             
    x = np.arange(len(folder_labels)) * group_spacing
    num_metrics = len(error_metrics)

    total_width = group_spacing * 0.9  # 90% of that spacing used by bars
    bar_width  = total_width / num_metrics

    fig, ax = plt.subplots(figsize=(8, 4))

    for i, metric in enumerate(error_metrics):
        offset = (i - (num_metrics-1)/2) * bar_width
        ax.bar(x + offset, all_errors[metric], bar_width, label=metric)

    ax.set_xticks(x, folder_labels)
    ax.margins(x=0)
    ax.set_xlim(x[0] - total_width/2, x[-1] + total_width/2)
    ax.set_ylabel("Error Value")
    ax.set_title("Voltage Error Metrics Comparison")
    ax.set_xticklabels(folder_labels, ha='right')
    ax.legend(loc = "best")
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig("Model_Voltage_PCT_Error_Metrics_Comparison.png", dpi=300)
    plt.show()

def plot_voltage_predictions_comparison(logs):
    """
    Plot ground truth and all predictions for the Voltage channel from multiple logs,
    overlaying predictions on the same plot for each cell.
    """
    voltage_channel_key = "0"  # Voltage channel key in the JSON

    # Plot the GT data once 
    first_log = logs[0][1] if logs else None
    if not first_log:
        print("No logs provided.")
        return
    
    cells = first_log.get("cells", {})
    cell_keys = list(cells.keys())

    num_cells = len(cell_keys)
    fig, axes = plt.subplots(num_cells, 1, figsize=(15, 5 * num_cells), sharex=True)

    # Make sure axes is always iterable
    if num_cells == 1:
        axes = [axes]

    # Plot ground truth for each cell first
    for row_idx, cell_key in enumerate(cell_keys):
        ax = axes[row_idx]
        gt = cells[cell_key][voltage_channel_key]["ground_truth"]
        ax.plot(gt, label="Ground Truth", color='red', linewidth=1)
        ax.set_title(f"Voltage @ Cell ({cell_key.replace('_', ', ')})", fontsize=10)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Voltage")
        ax.grid(False)

    # Overlay predictions from each folder with labels
    for folder, data in logs:
        label = make_label(folder, args.eval_metric)
        
        cells_data = data.get("cells", {})

        for row_idx, cell_key in enumerate(cell_keys):
            ax = axes[row_idx]
            pred = cells_data.get(cell_key, {}).get(voltage_channel_key, {}).get("prediction", None)
            if pred is not None:
                if "data" in folder.lower():
                    color = 'black'
                    linestyle = '--'
                    ax.plot(pred, label=label, color = color,  linestyle=linestyle, linewidth=1)
                else: 
                    ax.plot(pred, label=label, linestyle='--', linewidth=1)

    # Add legends (only once per subplot)
    for ax in axes:
        ax.legend(loc='upper right')

    plt.suptitle("Voltage Time Series Comparison Across Models", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = "Model_Cell_Plots_Voltage_Comparison.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)

def plot_voltage_predictions_comparison_unseen(logs):
    """
    Plot ground truth and all predictions for the Voltage channel from multiple logs,
    overlaying predictions on the same plot for each cell.
    """
    voltage_channel_key = "0"  # Voltage channel key in the JSON

    # Plot the GT data once 
    first_log = logs[0][1] if logs else None
    if not first_log:
        print("No logs provided.")
        return
    
    cells = first_log.get("cells", {})
    cell_keys = list(cells.keys())

    num_cells = len(cell_keys)
    fig, axes = plt.subplots(num_cells, 1, figsize=(15, 5 * num_cells), sharex=True)

    # Make sure axes is always iterable
    if num_cells == 1:
        axes = [axes]

    # Plot ground truth for each cell first
    for row_idx, cell_key in enumerate(cell_keys):
        ax = axes[row_idx]
        gt = cells[cell_key][voltage_channel_key]["ground_truth"]
        ax.plot(gt, label="Ground Truth", color='blue', linewidth=2)
        ax.set_title(f"Voltage @ Cell ({cell_key.replace('_', ', ')})", fontsize=10)
        ax.set_xlabel("Timestep")
        ax.set_xlim(200, len(gt))
        ax.set_ylabel("Voltage")
        ax.grid(False)

    # Overlay predictions from each folder with labels
    for folder, data in logs:
        # Extract label from folder name
        label = make_label(folder, args.eval_metric) 

        cells_data = data.get("cells", {})

        for row_idx, cell_key in enumerate(cell_keys):
            ax = axes[row_idx]
            pred = cells_data.get(cell_key, {}).get(voltage_channel_key, {}).get("prediction", None)
            if pred is not None:
                if "data" in folder.lower():
                    color = 'black'
                    linestyle = '--'
                    ax.plot(pred, label=label, color = color,  linestyle=linestyle, linewidth=1)
                else: 
                    ax.plot(pred, label=label, linestyle='--', linewidth=1)

    # Add legends (only once per subplot)
    for ax in axes:
        ax.legend(loc='upper right')

    plt.suptitle("Voltage Time Series Comparison Across Models", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = "Model_Cell_Plots_Voltage_Comparison_Unseen.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
  
## ======================================================================= ##
# == RUNNING THE SCRIPT == ##
## ======================================================================= ##


if __name__ == "__main__":
    logs = collect_training_logs()
    if logs:
        plot_combined_logs_training(logs)
        plot_combined_logs_testing(logs)
    else:
        print("No training logs found.")
    error_logs = collect_error_logs()
    if error_logs:
        plot_combined_errors(error_logs)
        #plot_combined_errors_unseen(error_logs)
    else:
        print('No error logs found.')
    results_logs = collect_json_results()
    if results_logs:
        #print(f'json results successfully loaded: {results_logs}')
        plot_results_avg(results_logs)
        plot_results_pct(results_logs)
    else: 
        print('No results metrics found.')
    cell_logs = collect_json_cells()
    if cell_logs:
        plot_voltage_predictions_comparison(cell_logs)
        #plot_voltage_predictions_comparison_unseen(cell_logs)
    else:
        print('No cell data found')