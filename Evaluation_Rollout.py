"""
Evaluating the results of training an FNO/PINO model on 2D AP simulations using the model as a simulator
=============================
Arguments: 
    - File path to load the datasets and trained model from. 
    - Resolution of the evaluation data
    - Time horizon to simulate out to. 
    - Whether to start from the initial input of the ground truth or select an initial window at random. 
"""

#Import required modules for constructing and training the model.
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
from neuralop.models import FNO
from neuralop.layers.embeddings import GridEmbeddingND
from AP_neuralop_utils import Trainer
from neuralop.training import AdamW
from AP_neuralop_utils import load_2D_AP_eval
from neuralop.utils import count_model_params
from neuralop.losses import LpLoss, H1Loss, HdivLoss, MSELoss, WeightedSumLoss
from AP_neuralop_utils import RMSELoss, APLoss, OperatorBackboneLoss
import random
import argparse
import os
from pathlib import Path
import re
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter 
#imports required for data logging
import json
import wandb
import matplotlib.ticker as ticker
import time


# Define the optional arguments to use when calling the operator
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Arguments for the data structure:
    parser.add_argument('-d', '--data-path', dest='data_path', required = True, type = str, help='File path for training and testing data results')
    parser.add_argument('-de', '--data-path-eval', dest='data_path_eval', required = True, type = str, help = 'File path for unseen dataset to evaluate on')
    parser.add_argument('-ne', '--n-eval', dest='n_eval', default = 100, help='Number of evaluation samples. Default is %(default)s')
    parser.add_argument('-er', '--eval-res', dest='eval_res', default = 200, type = int, help='Resolution of evaluation data (nxn). Default is n = %(default)s')
    parser.add_argument('-s', '--mesh-size', dest='mesh_size', default = 10, help='Size of the n x n mesh in cm. Default is n = %(default)s')
    parser.add_argument('-c', '--conmul', dest='conmul', default = 1.0, help='Conductivity multipler for simulated dataset. Default is n = %(default)s')

    # Arguments for the FNO and training parameters 
    parser.add_argument('-b', '--batch-size', dest='batch_size', default = 15, help='Batch size for evaluating the data. Default is %(default)s')
    parser.add_argument('-m', '--modes', dest='modes', default = 16, help='Number of modes to keep during Fourier transform. Default is %(default)s')
    parser.add_argument('-hc', '--hidden-channels', dest='hidden_channels', default = 32, help='Number of hidden channels for the FNO. Default is %(default)s')
    parser.add_argument('-em', '--eval-metric', dest='eval_metric', default = 'l2', help='Evaluation metric that was used for saving the best model. Default is %(default)s')
    parser.add_argument('-ch', '--channels', dest = 'ch', type = int, default = 2, help = 'Number of channels to use in training. Default is %(default)s (Voltage only). Change to 2 to include W (recovery current).' )
   
    # Arguments for simulation roll out:
    parser.add_argument('-ft', '--finetune', dest='fine_tune', action='store_true', help ='Boolean for inspecting results that were trained with a fine-tuning phase')
    parser.add_argument('-mw', '--moving-window', dest='moving_window', default = 5, type = int, help='Timestep moving window for input-output pair splitting. Default is %(default)s')
    parser.add_argument('-seen', '--seen', dest='seen', default = 5, type = float, help='Limit for seen timesteps in training. Default is %(default)s')
    parser.add_argument('-rnd', '--random', dest='random', action='store_true', help=' Select an initial window from the GT data to use for simulation at random.')
    parser.add_argument('-hor', '--horizon', dest='horizon', type = int, default = 200, help='Time horizon to run the simulator roll-out for. Default is all available timesteps in the GT data')
    

    args = parser.parse_args()

## ======================================================================= ##
# == DEFINING KEY FUNCTIONS FOR THE SIMULATOR  == ##
## ======================================================================= ##

#Default device for running the model evaluation (cuda)
device = 'cuda'

# Define the path to find the dataset to load from 
def get_data_root(custom_path: str):
    return (Path.cwd() / custom_path).resolve()

# Define function to use the model as a simulator to predict long roll out. 
def operator_simulator(model, initial_window, total_steps, window_size, device):
    """
    Run a block-predictor operator model (x->y) autoregressively in a feedback loop.
    
    Args:
        model: trained PINO/FNO model
        initial_window: [B, C, W, X, Y] - initial input window
        total_steps: int, number of future timesteps to simulate
        window_size: int, length of prediction window (same as training W)
        device: torch device ("cuda" or "cpu")
    
    Returns:
        rollout: [B, C, W + total_steps, X, Y]
    """
    state = initial_window.to(device)    # [B, C, W, X, Y]
    rollout = [state] 
    print(f"Initial State: {state.shape}")                   # keep trajectory, start with initial
    start_time = time.time()  # start timer
    steps_done = 0
    with torch.no_grad():
        while steps_done < total_steps:
            # predict next window
            y_pred = model(state) 
          
            # possibly trim if total_steps not multiple of W
            steps_left = total_steps - steps_done
            if steps_left < window_size:
                y_pred = y_pred[:, :, :steps_left]
                print(f"Next State: {y_pred.shape}")

            rollout.append(y_pred)

            # feed last predicted window as new state
            state = y_pred
            steps_done += y_pred.shape[2]

    # concatenate along time dimension and squeeze out the batch channel
    prediction = torch.cat(rollout, dim=2)
    prediction = prediction.squeeze(0)

    end_time = time.time()  # stop timer
    elapsed = end_time - start_time
    print(f"\nCompleted Rollout Simulation run in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes).")
    return prediction

# Define function to plot the errors during training
def plot_training_log(training_log_json, data_path, phase):

    with open(training_log_json, 'r') as file:
        data = json.load(file)
    '''
          # Use this section if the run was interupted 
    with open(training_log_json, 'r') as file:
            data = [json.loads(line) for line in file if line.strip()]

    with open(training_log_json, 'w') as file:
        json.dump(data, file, indent=2)
    
    '''
    # Extract training log data for visualisation
    epochs = [entry["epoch"] for entry in data]
    train_err = [entry["train_err"] for entry in data]
    avg_loss = [entry["avg_loss"] for entry in data]
    train_time =  [entry["epoch_train_time"] for entry in data]

    # Load the testing errors throughout the training process
    h1 = [next((v for k, v in entry.items() if re.search(r"_h1$", k)), None) for entry in data]
    l2 = [next((v for k, v in entry.items() if re.search(r"_l2$", k)), None) for entry in data]
    mse = [next((v for k, v in entry.items() if re.search(r"_mse$", k)), None) for entry in data]
    rmse = [next((v for k, v in entry.items() if re.search(r"_rmse$", k)), None) for entry in data]
    phys_ap = [next((v for k, v in entry.items() if re.search(r"_ap_phys$", k)), None) for entry in data]
    combined = [next((v for k, v in entry.items() if re.search(r"_combined$", k)), None) for entry in data]
    oploss = [next((v for k, v in entry.items() if re.search(r"_op$", k)), None) for entry in data]
    ftloss = [next((v for k, v in entry.items() if re.search(r"_ft$", k)), None) for entry in data]

    # Plot the results:  
    # 1: Training and evaluation (testing) losses for the epochs 
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, avg_loss, label="Training Loss ")
    
    #if any(h1): plt.plot(epochs, h1, label="Testing (Evaluation) Loss - h1 ", color = 'green', linestyle = 'dashdot')
    #if any(l2): plt.plot(epochs, l2, label="Testing (Evaluation) Loss - l2 ", color = 'green')
    plt.plot(epochs, mse, label="Testing Loss - mse ")
    #if any(phys_ap): plt.plot(epochs, phys_ap, label="Testing (Evaluation) Loss - ap_phys ", color = 'red')
    #if any(combined): plt.plot(epochs, combined, label="Testing (Evaluation) Loss - combined ", color = 'red', linestyle = 'dashdot')
    #if any(oploss): plt.plot(epochs, combined, label="Testing (Evaluation) Loss - operator ", color = 'purple')
    #if any(ftloss): plt.plot(epochs, ftloss, label="Testing (Evaluation) Loss - finetune ", color = 'purple', linestyle = 'dashdot')

    plt.xlabel("Epoch")
    plt.ylabel("Model Losses")
    plt.title("Training/Testing Losses over Epochs")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(data_path + "/Training_Testing_Losses_" + phase + ".png"))
   # plt.show()


    '''
    # 2: Training Time  
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_time, color = 'black')

    plt.xlabel("Epoch")
    plt.ylabel("Training Time")
    plt.title("Model Training Time")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(data_path + "/Training_Time_" + phase + ".png"))
   # plt.show()
   '''
    
    # 2: Compute and return NET training time and overall/average rmse value on the training data:
    net_training_time = sum(train_time)
    # Filter out None values for mse/rmse
    valid_mse = [(i, v) for i, v in enumerate(mse) if v is not None]
    valid_rmse = [(i, v) for i, v in enumerate(rmse) if v is not None]

    min_mse, min_mse_epoch = (None, None)
    min_rmse, min_rmse_epoch = (None, None)

    if valid_mse:
        min_mse_idx, min_mse = min(valid_mse, key=lambda x: x[1])
        min_mse_epoch = epochs[min_mse_idx]
    if valid_rmse:
        min_rmse_idx, min_rmse = min(valid_rmse, key=lambda x: x[1])
        min_rmse_epoch = epochs[min_rmse_idx]

    # === Print summary ===
    print(f"\nNet training time: {net_training_time:.2f} seconds ({net_training_time/60:.2f} minutes)")
    if min_mse is not None:
        print(f" Minimum MSE: {min_mse:.6f} at epoch {min_mse_epoch}")
    if min_rmse is not None:
        print(f" Minimum RMSE: {min_rmse:.6f} at epoch {min_rmse_epoch}\n")

    # === Save summary to JSON ===
    summary = {
        "phase": phase,
        "net_training_time_sec": net_training_time,
        "net_training_time_min": net_training_time / 60,
        "min_mse": min_mse,
        "min_mse_epoch": min_mse_epoch,
        "min_rmse": min_rmse,
        "min_rmse_epoch": min_rmse_epoch
    }

    summary_path = os.path.join(data_path, f"Training_Summary_{phase}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Training summary saved to: {summary_path}")

# Define function to extract relevant information from the dataset
def info_extraction(dataset_info):
    print('\n### --------- ###\n')
    #Use the dataset info to extract the relevant parameters for embedding and training set up 
    target_name = f"dataset_info_{args.eval_res}_{args.conmul}.txt"
    dataset_info = os.path.join(args.data_path_eval, target_name)
    print(dataset_info)
    with open(dataset_info, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('Grid_resolution'):
                numbers = re.findall(r'\d+\.?\d*', line)
                if numbers:
                    dx = float(numbers[0])
                    print(f"dx: {dx}")
                    dy = float(numbers[0])
                    print(f"dy: {dy}")
            if line.startswith('Timestep resolution'):
                numbers = re.findall(r'\d+', line)
                if numbers:
                    delta_t = int(numbers[0])
                    print(f"delta_t: {delta_t} ms")
            if line.startswith('Evaluation data shape:'):
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        num_timesteps = int(numbers[1])
                        print(f'Number of evaluation time frames:{num_timesteps}')
            if line.startswith('Training data shapes:'):
                numbers = re.findall(r'\d+', line)
                if numbers:
                    n_train = int(numbers[0])
                    print(f"n_train: {n_train}")
                    train_res = int(numbers[3])
                    print(f"training_resolution: {train_res}")
            if line.startswith("Input-Output pairs"):
                numbers = re.findall(r'\d+', line)
                if numbers:
                    time_frames = numbers[0]
                    print(f"Time boundary: {time_frames} frames")
            if line.startswith("Resting potential (E_rest) = "):
                # Match numbers (including negative and decimal)
                numbers = re.findall(r'-?\d+\.\d+', line)
                if numbers and len(numbers) >= 2:
                    V_rest = float(numbers[0]) 
                    V_amp = float(numbers[1])
                    print(f"V_rest = {V_rest}, V_amp = {V_amp}")
        print('\n### --------- ###\n')
        return  dx, dy, delta_t, time_frames, num_timesteps
        
'''
RESULTS:
Plot the training and evaluation losses.

Re-load the best performing model and use this model as a simulator on the evaluation set: 
    - Extract the first (or random) sample from the dataset and use the model to predict the next window of timesteps. 
    - Using the output of the model, predict the next set of timesteps in a feedback loop until the full timesteps of the prediction horizon have been simulated. 
   
Following the use of the model as a simulator, compare the prediction to the ground truth:
    - Return information about the best/worst performing cells and timesteps.
    - Plot the sample visualisations - 3 selected samples chosen to compare GT with prediction across all models 
    - Plot the APDs - 3 selected samples chosen to compare GT with prediction across all models 
'''

## ======================================================================= ##
# == PLOTTING THE LOSSES THROUGHOUT TRAINING & TESTING PROCESS == ##
## ======================================================================= ##
if args.fine_tune: 
    
    # Plotting the fine tuning phase of training 
    training_log_json_2 = os.path.join(args.data_path, "training_log_phase_2.json")
    phase = "fine_tuning_phase"

    # Convert the json training log file into a valid file to read in the results analysis:
    
    with open(training_log_json_2, 'r') as file:
        data = [json.loads(line) for line in file if line.strip()]

    with open(training_log_json_2, 'w') as file:
        json.dump(data, file, indent=2)


    plot_training_log(training_log_json_2, args.data_path, phase)

else:
    # Plotting the training
    training_log_json_1 = os.path.join(args.data_path, "training_log.json")
    phase = ""
    plot_training_log(training_log_json_1, args.data_path, phase) 


## ===================================================================================== ##
# == RUNNING THE BEST PERFORMING MODEL AS A SIMULATOR FOR TEMPORAL ROLL-OUT == ##
## ===================================================================================== ##

## ----------------------------------------------------------------------- ##
#  LOADING THE EVALUATION DATASET
## ----------------------------------------------------------------------- ##

# Load the evaluation datasets to test the model on:
custom_path = args.data_path_eval
example_data_root = get_data_root(custom_path)

#Default values for testing and training (if no dataset info file is available)
n_eval = args.n_eval
#Use the dataset info to extract the relevant parameters for embedding and training set up 
target_name = f"dataset_info_{args.eval_res}_{args.conmul}.txt"
dataset_info = os.path.join(args.data_path_eval, target_name)

# Extract information from the dataset 
dx, dy, delta_t, time_frames, num_timesteps = info_extraction(dataset_info)

# Setting the number of timesteps seen during the training phase
print(f'Total number of frames in evaluation set: {num_timesteps}') 

'''

# Load the dataset (using the new loading function defined for the dataset): 
# No batching of the evaluation dataset (i.e use a single batch of the full dataset)
eval_loader, data_processor = load_2D_AP_eval(
        n_eval=n_eval, batch_size=int(num_timesteps), data_root=example_data_root, eval_resolution = H, dataset_name = '2D_AP', cm_eval = args.conmul,
         encode_input = False, encode_output = False,
)
data_processor = data_processor.to(device)
full_dataset = eval_loader.dataset


'''

# Establish the embedding and boundaries for the inclusion of the time channel (Adapting the spatial embeddings for the grid resolutions):
print('\n### --------- ###\n')
print(f"Mesh size: {args.mesh_size} cm x {args.mesh_size} cm")
time_boundary = float(delta_t) * (float(time_frames) -1 )
print(f"Sample Time Boundary: {time_frames} frames with spacing {delta_t} ms =  {time_boundary} ms ")
embedding = GridEmbeddingND(in_channels=args.ch, dim=3, grid_boundaries=[[0,float(time_boundary)], [0, args.mesh_size], [0, args.mesh_size]])
print("Embedding Grid Boundaries =", embedding.grid_boundaries)


## ----------------------------------------------------------------------- ##
#  LOADING THE PRE-TRAINED MODEL
## ----------------------------------------------------------------------- ##
print('--------------------------------------------------')
best_model_path = os.path.join(args.data_path, f'best_model_state_dict.pt')
print(f"LOADING BEST PERFORMING MODEL FROM: {best_model_path}")
print('--------------------------------------------------')
json_log_path_best=os.path.join(args.data_path, "Best_Model_Results.json")
sys.stdout.flush()


#embedding = GridEmbeddingND(in_channels=args.ch, dim=3, grid_boundaries=[[0,float(args.moving_window)], [0, H], [0, W]])
#print("grid_boundaries =", embedding.grid_boundaries)
model = FNO(n_modes=(8, 16, 16),
             in_channels=args.ch, 
             out_channels=args.ch,
             hidden_channels=32, 
             projection_channel_ratio=2,
             positional_embedding = embedding)
model = model.to(device)
model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=False))
model.eval()


## ================================================================================= ##
# Load the ground truth data for comparison: 
folder_path = args.data_path_eval
target_dataset = f"2D_AP_eval_full_{args.eval_res}_{args.conmul}.pt"

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename == target_dataset:
        file_path = os.path.join(folder_path, filename)
        tensor = torch.load(file_path, map_location="cuda")
        print('--------------------------------------------------')
        print(f"LOADING EVALUATION DATASET: {filename}")
        print('--------------------------------------------------')

y_true_full = tensor
print(f"Ground truth shape: {y_true_full.shape}")

## ================================================================================= ##

## ----------------------------------------------------------------------- ##
#  RUNNING THE SIMULATOR
## ----------------------------------------------------------------------- ##

## ================================================================================= ##

if args.random:
    print('--------------------------------------------------')
    print(f"Selecting input window at random")
    # Extract the initial state at random from the full dataset 
    #rand_idx = random.randint(0, num_timesteps - 1)
    rand_idx = 5
    window_size = int(time_frames)
    print(f"Input index: {rand_idx}")
    input_window = rand_idx + window_size
    print(f"Input window: Timesteps {rand_idx} to {input_window}")
    
    print(f"Ground truth shape: {y_true_full.shape}")
    x0 = y_true_full[:, rand_idx:rand_idx + int(time_frames), :, :].unsqueeze(0).to(device)
    
    # Perform the simulation roll out from the initial sample 
    print(f"Shape of input sample: {x0.shape}")

    end_t = rand_idx + x0.shape[2]
    if args.horizon > num_timesteps:
        total_steps = int(args.horizon) - end_t
    else: 
        total_steps = num_timesteps - end_t
    print(f"Total timesteps left to predict: {total_steps}")
    y_pred_full = operator_simulator(model, x0, total_steps, window_size, device)
    print(f"Simulator output shape: {y_pred_full.shape}")
    print('--------------------------------------------------')

else: 
    print(f"Selecting input window from start of trajectory")
    print('--------------------------------------------------')
    # Extract the initial state from the full data structure
    idx = 0
    window_size = int(time_frames)
    print(f"Input index: {idx}")
    input_window = idx + window_size
    print(f"Input window: Timesteps {idx} to {input_window}")
    
    print(f"Ground truth shape: {y_true_full.shape}")
    x0 = y_true_full[:, idx:idx + int(time_frames), :, :].unsqueeze(0).to(device)
    
    # Perform the simulation roll out from the initial sample 
    print(f"Shape of input sample: {x0.shape}")

    end_t = idx + x0.shape[2]
    if args.horizon > num_timesteps:
        total_steps = int(args.horizon) - end_t
    else: 
        total_steps = num_timesteps - end_t
    print(f"Total timesteps left to predict: {total_steps}")
    y_pred_full = operator_simulator(model, x0, total_steps, window_size, device)
    print(f"Simulator output shape: {y_pred_full.shape}")
    print('--------------------------------------------------')



## ----------------------------------------------------------------------- ##
#  COMPARING THE SIMULATOR OUTPUT AND GROUND TRUTH DATA
## ----------------------------------------------------------------------- ##

# Calculate the signed and absolute errors between the prediction and ground truth
if args.random:
    # Length of simulator output
    pred_len = y_pred_full.shape[1]

    # Make sure GT slice doesn’t go past the end of the trajectory
    start_t = rand_idx
    end_t = min(start_t + pred_len, y_true_full.shape[1])
    y_true_aligned = y_true_full[:, start_t:end_t]

    # Trim prediction to match GT slice
    y_pred_aligned = y_pred_full[:, :y_true_aligned.shape[1]]

    print("Simulator aligned shape:", y_pred_aligned.shape)
    print("Ground truth aligned shape:", y_true_aligned.shape)
else:


    min_t = min(y_pred_full.shape[1], y_true_full.shape[1])
    y_pred_aligned = y_pred_full[:, :min_t, ...]
    y_true_aligned = y_true_full[:, :min_t, ...]

    print("Pred aligned:", y_pred_aligned.shape)
    print("True aligned:", y_true_aligned.shape)

    
    #y_pred_aligned = y_pred_full[:, :, :y_true_full.shape[1]] 
    #print("Simulator aligned shape:", y_pred_full.shape) # should match the ground truth shape
    

    #Rename the GT data for alignment purposes: 
    y_true_aligned = y_true_full


signed_error = y_pred_aligned - y_true_aligned 
abs_error = torch.abs(signed_error)
mse_per_timestep_list = []
rmse_per_timestep_list = []
# Initialize analysis results

analysis = {}

# Error calculation per channel
num_channels = y_true_full.shape[0]
for c in range(num_channels):
    channel_name = f"Channel {c} (0 = Voltage, 1 = Recovery Current)"

    # Errors for this channel
    channel_signed_error = signed_error[c, :, :, :]  # [T, X, Y]
    channel_abs_error = abs_error[c, :, :, :]        # [T, X, Y]

    # Metric 1: Mean Absolute Error
    avg_mae = channel_abs_error.mean().item()

    # Metric 2: Mean Absolute Signed Error (bias)
    mean_abs_signed_error = channel_signed_error.abs().mean().item()

    # Metric 3: Max Absolute Signed Error
    max_abs_signed_error = channel_signed_error.abs().max().item()

    # Metric 4: Per-cell average error
    error_per_cell = channel_abs_error.mean(dim=(0, 1))  # [X, Y]
    avg_cell = error_per_cell.mean().item()

    # Metric 5: Top 10 worst cells
    top10 = torch.topk(error_per_cell.flatten(), k=10)
    avg_top10_cell = top10.values.mean().item()

   # Metric 6: Percentage of individual errors above average MAE
    total_elements = channel_abs_error.numel()
    threshold_exceed_count = (channel_abs_error > avg_mae).sum().item()
    threshold_exceed_pct = (threshold_exceed_count / total_elements) * 100

    # Metric 7: Percentage of cells with average error above average cell error
    total_cells = error_per_cell.numel()
    num_cells_above_threshold = (error_per_cell > avg_cell).sum().item()
    num_cells_above_threshold_pct = (num_cells_above_threshold / total_cells) * 100

    # Metric 8: Worst-case (max abs error)
    max_error = channel_abs_error.max().item()
    max_idx = torch.nonzero(channel_abs_error == max_error, as_tuple=False)[0]
    worst_case = {
        #"Batch": int(max_idx[0]),
        "Timestep": int(max_idx[0]),
        "Cell [x, y]": [int(max_idx[1]), int(max_idx[2])],
        "Absolute Error": max_error
    }

    # Metric 9: Best-case (min abs error)
    min_error = channel_abs_error.min().item()
    min_idx = torch.nonzero(channel_abs_error == min_error, as_tuple=False)[0]
    best_case = {
        #"Batch": int(min_idx[0]),
        "Timestep": int(min_idx[0]),
        "Cell [x, y]": [int(min_idx[1]), int(min_idx[2])],
        "Absolute Error": min_error
    }

    # Metric 10: Best/Worst cell overall across time
    error_map = channel_abs_error.mean(dim=0)  # [X, Y]
    best_overall_error = error_map.min().item()
    worst_overall_error = error_map.max().item()

    best_flat = torch.argmin(error_map)
    worst_flat = torch.argmax(error_map)

    best_x, best_y = divmod(best_flat.item(), error_map.shape[1])
    worst_x, worst_y = divmod(worst_flat.item(), error_map.shape[1])

    best_worst_over_time = {
        "Best Cell [x, y]": [best_x, best_y],
        "Best Cell Avg Error": best_overall_error,
        "Worst Cell [x, y]": [worst_x, worst_y],
        "Worst Cell Avg Error": worst_overall_error
    }


    # Metric 11: MSE across batch and cells for each timestep
    mse_per_timestep = (channel_signed_error ** 2).mean(dim= (1, 2))  # [T]
    mse_per_timestep_list.append(mse_per_timestep.cpu().numpy().tolist())

    # Metric 12: RMSE across batch and cells for each timestep AND As an overall metric
    
    # Metric 12: RMSE across batch and cells for each timestep AND As an overall metric
    # Compute RMSE per timestep
    rmse_per_timestep = torch.sqrt((channel_signed_error ** 2).mean(dim=(1, 2)))  # [T]
    rmse_per_timestep_list.append(rmse_per_timestep.cpu().numpy().tolist())

    # Compute overall RMSE over the whole evaluation set
    overall_rmse = torch.sqrt((channel_signed_error ** 2).mean())
    print("Overall RMSE:", overall_rmse.item())
        
    
    # Store everything
    analysis[channel_name] = {
        "Avg Mean Absolute Error": avg_mae,
        "Average Cell Error": avg_cell,
        "Average Error of Top 10 Worst Cells": avg_top10_cell,
        "Mean Signed Error (Bias)": mean_abs_signed_error,
        "Max Signed Error": max_abs_signed_error,
        "Pct Errors > Avg MAE": threshold_exceed_pct,
        "Pct Cells with Avg Error > Avg Cell Error": num_cells_above_threshold_pct,
        "Worst Case Results": worst_case,
        "Best Case Results": best_case,
        "Best/Worst Cells Over Time": best_worst_over_time,
        "Overall RMSE:" : overall_rmse.item()
        #"Per-Timestep MSE": mse_per_timestep_list
    }

# Collate errors from the model:
avg_results = {}
avg_results.update(analysis)

with open(json_log_path_best, "w") as f:
    json.dump(avg_results, f, indent=2)

print('--------------------------------------------------')
print("MODEL EVALUATION RESULTS:")
for ch_key, ch_result in analysis.items():
    print(f"\n{ch_key}:")
    for k, v in ch_result.items():
        print(f"  {k}: {v}")
print('--------------------------------------------------')


## ======================================================================= ##
# == VISUALISATION ANALYSIS == ##
## ======================================================================= ##

# Change device to cpu for plotting
device = 'cpu'
if args.ch == 2:
    channels = [0,1]
    channel_names = ["Voltage (V)", "Recovery (W)"]
elif args.ch == 1:
    channels = [0]
    channel_names = ["Voltage (V)"]


## ----------------------------------------------------------------------- ##
# TIME SERIES PLOTTING OF V AND W FOR BEST-WORST PERFORMING CELLS
## ----------------------------------------------------------------------- ##

# extract the cells from the earlier error calculations: 
cells = [
    (best_x, best_y),
    (worst_x, worst_y)
]

# Remove batch dimension to ease visualisation
gt = y_true_full 
pred = y_pred_full

cell_data = {
    "model_name": args.data_path,
    "channels": {
        "0": "Voltage",
        "1": "Recovery"
    },
    "cells": {}  # Will store results keyed by cell tuple
}

# For random index start points - adjust the x axiss to match the ground truth
if args.random:
    # Prediction rollout length
    pred_len = pred.shape[1]  # e.g., 431
    x_pred = np.arange(rand_idx, rand_idx + pred_len)  # global timesteps for prediction
else:
    pred_len = pred.shape[1]
    x_pred = np.arange(pred_len)

# Ground truth full x-axis
x_gt_full = np.arange(gt.shape[1])

# Save the time series for the cells (later used for model comparison plots)
for ch in [0, 1]:
    for cell_x, cell_y in cells:
        
        cell_key = f"{cell_x}_{cell_y}"
        
        gt_series = gt[ch, :, cell_x, cell_y].detach().cpu().numpy().tolist()
        pred_series = pred[ch, :, cell_x, cell_y].detach().cpu().numpy().tolist()
     
               # if args.random:
               
               #     gt_series = gt[ch, rand_idx:rand_idx + pred_len, cell_x, cell_y].detach().cpu().numpy().tolist()
               # else:
               # 
               #     gt_series = gt[ch, :, cell_x, cell_y].detach().cpu().numpy().tolist()

               # pred_series = pred[ch, :, cell_x, cell_y].detach().cpu().numpy().tolist()
       
        # Initialize if cell not already present
        if cell_key not in cell_data["cells"]:
            cell_data["cells"][cell_key] = {}

        cell_data["cells"][cell_key][f"{ch}"] = {
            "ground_truth": gt_series,
            "prediction": pred_series
        }

# Save to JSON
save_path = os.path.join(args.data_path, "best_worst_cell_time_series.json")
with open(save_path, "w") as f:
    json.dump(cell_data, f, indent=2)


# Plot the full time series (Voltage only)
titles = ["Best Cell", "Worst Cell"]

if args.seen != 0.0:
    # Mark where the initial input frames end in the sequential roll-out simulation 
    if args.random: 
        x_line = rand_idx + args.seen
    else: 
        x_line = args.seen  # the x-coordinate where you want the line
    line_label = "Unseen Time Horizon"  # text label for the line

# Create a figure with 2 vertical subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

for i, (cell_x, cell_y) in enumerate(cells):
    
    ax = axes[i]

    # Full ground truth (plot the entire dataset)
    gt_full_series = gt[0, :, cell_x, cell_y].detach().cpu().numpy()
    # Prediction rollout (starting from the random initial window)
    pred_series = pred[0, :, cell_x, cell_y].detach().cpu().numpy()

    print("x_gt_full shape:", x_gt_full.shape)
    print("pred_series shape:", pred_series.shape)


    # Plot full GT and prediction rollout
    ax.plot(x_gt_full, gt_full_series, label='Ground Truth (full)', color='blue', linewidth=2)
    if args.random:
        x_pred_plot = np.arange(rand_idx, rand_idx + pred_series.shape[0])
        ax.plot(x_pred_plot, pred_series, label='Prediction', color='red', linewidth=2, linestyle='--')
    else:
        ax.plot(x_gt_full, pred_series, label='Prediction', color='red', linewidth=2, linestyle='--')


    ax.set_title(f"{titles[i]} - Cell ({cell_x}, {cell_y}) - Channel 0 (Voltage)")
    ax.set_ylabel("Voltage (AU)")
    ax.grid(False)

    # Optional: mark unseen horizon
    if args.seen and not args.random:
        ax.axvline(x=args.seen, color='black', linestyle=':', linewidth=2)
        ax.text(args.seen + 2, ax.get_ylim()[1]*0.95, '', color='green', rotation=90, verticalalignment='top')

    # Optional: mark initial window
    if args.random:
        ax.axvspan(rand_idx, rand_idx + window_size, color='gray', alpha=0.2, label='Initial Window')

    ax.legend()

#Common x-label
#axes[-1].set_xlabel("Time step (AU)")
# Convert tick labels from frame index → milliseconds
axes[-1].set_xlabel("Time step (ms)")
for ax in axes:
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: int(x * int(delta_t))))
plt.tight_layout()
save_path = os.path.join(args.data_path, 'Best_Worst_Voltage_Cell_Plots.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.close(fig)


## ----------------------------------------------------------------------- ##
# TIME SERIES PLOTTING OF V AND W FOR SELECTED CELLS FOR COMPARISON 
## ----------------------------------------------------------------------- ##

# Pick specific grid cells to monitor in the mesh - the centre, the bottom left and the top right cell. 

centre = int(int(args.eval_res) / 2)
quarter = int(int(args.eval_res) / 4)
three_quarter = int(int(args.eval_res) * (3/4))

cells = [[quarter, quarter], [centre, centre], [three_quarter, three_quarter]]

print(f"SELECTED CELLS: {cells}")


cell_data = {
    "model_name": args.data_path,
    "channels": {
        "0": "Voltage",
        "1": "Recovery"
    },
    "cells": {}  # Will store results keyed by cell tuple
}

# Save the time series for the cells (later used for model comparison)
for ch in [0, 1]:
    for cell_x, cell_y in cells:
        cell_key = f"{cell_x}_{cell_y}"

        gt_series = gt[ch, :, cell_x, cell_y].tolist()
        pred_series = pred[ch, :, cell_x, cell_y].tolist()

        # Initialize if cell not already present
        if cell_key not in cell_data["cells"]:
            cell_data["cells"][cell_key] = {}

        cell_data["cells"][cell_key][f"{ch}"] = {
            "ground_truth": gt_series,
            "prediction": pred_series
        }

# Save to JSON
save_path = os.path.join(args.data_path, "cell_time_series.json")
with open(save_path, "w") as f:
    json.dump(cell_data, f, indent=2)


# Plot the full time series (Voltage only)
if args.seen != 0.0:
    # Mark where the initial input frames end in the sequential roll-out simulation 
    if args.random: 
        x_line = rand_idx + args.seen
    else: 
        x_line = args.seen  # the x-coordinate where you want the line
    line_label = "Unseen Time Horizon"  # text label for the line


# Create a figure with 3 vertical subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

for i, (cell_x, cell_y) in enumerate(cells):
    
    ax = axes[i]

    # Full ground truth (plot the entire dataset)
    gt_full_series = gt[0, :, cell_x, cell_y].detach().cpu().numpy()
    # Prediction rollout (starting from the random initial window)
    pred_series = pred[0, :, cell_x, cell_y].detach().cpu().numpy()

    # Plot full GT and prediction rollout
    ax.plot(x_gt_full, gt_full_series, label='Ground Truth (full)', color='blue', linewidth=2)
    if args.random:
        x_pred_plot = np.arange(rand_idx, rand_idx + pred_series.shape[0])
        ax.plot(x_pred_plot, pred_series, label='Prediction', color='red', linewidth=2, linestyle='--')
    else:
        ax.plot(x_gt_full, pred_series, label='Prediction', color='red', linewidth=2, linestyle='--')
    
    ax.set_title(f"Voltage @ - Cell ({cell_x}, {cell_y}) - Channel 0 (Voltage)")
    ax.set_ylabel("Voltage (AU)")
    ax.grid(False)

    # Optional: mark unseen horizon
    if args.seen and not args.random:
        ax.axvline(x=args.seen, color='black', linestyle=':', linewidth=2)
        ax.text(args.seen + 2, ax.get_ylim()[1]*0.95, '', color='green', rotation=90, verticalalignment='top')

    # Optional: mark initial window
    if args.random:
        ax.axvspan(rand_idx, rand_idx + window_size, color='gray', alpha=0.2, label='Initial Window')

    ax.legend()


# Common x-label
#axes[-1].set_xlabel("Time step (AU)")
# Convert tick labels from frame index → milliseconds
axes[-1].set_xlabel("Time step (ms)")
for ax in axes:
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: int(x * int(delta_t))))
plt.tight_layout()
save_path = os.path.join(args.data_path, 'Selected_Voltage_Cell_Plots.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.close(fig)




## ----------------------------------------------------------------------- ##
# SAMPLE PLOTS OF GT, PREDICTION, AND ERROR (with colorbars)
## ----------------------------------------------------------------------- ##

def plot_selected_frames_with_input(x, gt, pred, save_dir, selected_frames=[0]):
    """
    Plots selected timesteps of INPUT, GT, PRED, and ERROR side-by-side.

    Args:
        x  : torch.Tensor [1, T_in, H, W] or [B, T_in, H, W] (model input sequence)
        gt : torch.Tensor [1, T, H, W] or [B, T, H, W]      (ground truth)
        pred : torch.Tensor [1, T, H, W] or [B, T, H, W]    (predictions)
        save_dir : directory to save output image
        selected_frames : list of timesteps to visualize
    """

    os.makedirs(save_dir, exist_ok=True)

    # Convert tensors to numpy
    voltage_in = x[0].detach().cpu().numpy()       # [T_in, H, W]
    voltage_gt = gt[0].detach().cpu().numpy()      # [T, H, W]
    voltage_pred = pred[0].detach().cpu().numpy()  # [T, H, W]
    voltage_err = voltage_gt - voltage_pred        # [T, H, W]

    # Color scale limits
    vmin = min(voltage_in.min(), voltage_gt.min(), voltage_pred.min())
    vmax = max(voltage_in.max(), voltage_gt.max(), voltage_pred.max())
    err_abs = np.abs(voltage_err).max()

    num_frames = len(selected_frames)

    # 4 rows: Input, GT, Pred, Error
    fig, axs = plt.subplots(4, num_frames, figsize=(4 * num_frames, 12))
    if num_frames == 1:
        axs = np.expand_dims(axs, axis=1)

    row_labels = ["Input", "GT", "Pred", "Error"]

    for col, frame in enumerate(selected_frames):
        in_frame = voltage_in[frame-int(args.moving_window)]
        gt_frame = voltage_gt[frame]
        pred_frame = voltage_pred[frame]
        err_frame = voltage_err[frame]

        im0 = axs[0, col].imshow(in_frame, cmap='viridis', vmin=vmin, vmax=vmax)
        axs[0, col].set_title(f"Input - Timestep {frame-int(args.moving_window)}", fontsize=10)
        im1 = axs[1, col].imshow(gt_frame, cmap='viridis', vmin=vmin, vmax=vmax)
        axs[1, col].set_title(f"GT - Timestep {frame}", fontsize=10)
        im2 = axs[2, col].imshow(pred_frame, cmap='viridis', vmin=vmin, vmax=vmax)
        axs[2, col].set_title(f"Pred - Timestep {frame}", fontsize=10)
        im3 = axs[3, col].imshow(err_frame, cmap='bwr', vmin=-err_abs, vmax=err_abs)
        axs[3, col].set_title("Error (GT - Pred)", fontsize=10)

        for row in range(4):
            axs[row, col].axis('off')

    # Add row labels on the left
    for row in range(4):
        axs[row, 0].text(
            -0.3, 0.5, row_labels[row],
            fontsize=12, fontweight='bold',
            rotation='vertical', ha='center', va='center',
            transform=axs[row, 0].transAxes
        )

    # Adjust spacing
    plt.subplots_adjust(wspace=0.05, hspace=0.35, top=0.9, bottom=0.05)

    # Add colorbars
    for im, row_axes in zip([im0, im1, im2, im3], axs):
        cbar = fig.colorbar(im, ax=row_axes.ravel().tolist(),
                            fraction=0.02, pad=0.04, orientation='vertical')
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle("Input, Ground Truth, Prediction, and Error (Selected Frames)", fontsize=16)
    save_path = os.path.join(save_dir, "Voltage_Samples_With_Input.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved input+GT+prediction visualization to {save_path}")

def plot_selected_frames(gt, pred, save_dir, selected_frames=[0, 50, 100, 150, 199]):
    """
    Plots selected timesteps of GT, prediction, and error side-by-side
    with row labels and adjustable spacing between rows.
    """

    # Convert tensors to numpy arrays
    voltage_gt = gt[0].detach().cpu().numpy()      # [T, H, W]
    voltage_pred = pred[0].detach().cpu().numpy()  # [T, H, W]
    voltage_err = voltage_gt - voltage_pred        # [T, H, W]

    # Consistent color scales
    vmin = min(voltage_gt.min(), voltage_pred.min())
    vmax = max(voltage_gt.max(), voltage_pred.max())
    err_abs = np.abs(voltage_err).max()

    num_frames = len(selected_frames)

    # Create figure grid: 3 rows (GT, Pred, Err) × N columns (frames)
    # Disable constrained_layout since we will manage spacing manually
    fig, axs = plt.subplots(3, num_frames, figsize=(4*num_frames, 10))

    if num_frames == 1:
        axs = np.expand_dims(axs, axis=1)  # ensure consistent indexing

    row_titles = ["Ground Truth", "Prediction", "Error (GT - Pred)"]
    short_labels = ["GT", "Pred", "Error"]

    for col, frame in enumerate(selected_frames):
        gt_frame = voltage_gt[frame]
        pred_frame = voltage_pred[frame]
        err_frame = voltage_err[frame]

        # Plot data
        im0 = axs[0, col].imshow(gt_frame, cmap='viridis', vmin=vmin, vmax=vmax)
        axs[0, col].set_title(f"Timestep {frame}", fontsize=10)
        im1 = axs[1, col].imshow(pred_frame, cmap='viridis', vmin=vmin, vmax=vmax)
        im2 = axs[2, col].imshow(err_frame, cmap='bwr', vmin=-err_abs, vmax=err_abs)

        for row in range(3):
            axs[row, col].axis('off')

    # Label rows (leftmost column)
    for row in range(3):
        axs[row, 0].text(
            -0.3, 0.5,               # position relative to image
            #f"{short_labels[row]}\n{row_titles[row]}",
            f"{short_labels[row]}",
            fontsize=12,
            fontweight="bold",
            rotation='vertical',
            ha='center',
            va='center',
            transform=axs[row, 0].transAxes
        )

    # Adjust spacing manually
    plt.subplots_adjust(wspace=0.05, hspace=0.35, top=0.9, bottom=0.05)

    # Add colorbars (shared per row)
    for row, im in enumerate([im0, im1, im2]):
        cbar = fig.colorbar(
            im, ax=axs[row, :].ravel().tolist(),
            fraction=0.02, pad=0.04, orientation='vertical'
        )
        cbar.ax.tick_params(labelsize=8)

    # Title
    fig.suptitle("Selected Voltage Frames: GT, Prediction, and Error", fontsize=16)

    # Save figure
    save_path = os.path.join(save_dir, "Voltage_Samples.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved selected frame visualization to {save_path}")

def plot_selected_frames(gt, pred, save_dir, selected_frames=[0, 50, 100, 150, 199]):
    """
    Plots selected timesteps of GT, prediction, and error side-by-side
    with row labels and adjustable spacing between rows.
    """

    # Convert tensors to numpy arrays
    voltage_gt = gt[0].detach().cpu().numpy()      # [T, H, W]
    voltage_pred = pred[0].detach().cpu().numpy()  # [T, H, W]
    voltage_err = voltage_gt - voltage_pred        # [T, H, W]

    # Consistent color scales
    vmin = min(voltage_gt.min(), voltage_pred.min())
    vmax = max(voltage_gt.max(), voltage_pred.max())
    err_abs = np.abs(voltage_err).max()

    num_frames = len(selected_frames)

    # Create figure grid: 3 rows (GT, Pred, Err) × N columns (frames)
    # Disable constrained_layout since we will manage spacing manually
    fig, axs = plt.subplots(3, num_frames, figsize=(4*num_frames, 10))

    if num_frames == 1:
        axs = np.expand_dims(axs, axis=1)  # ensure consistent indexing

    row_titles = ["Ground Truth", "Prediction", "Error (GT - Pred)"]
    short_labels = ["GT", "Pred", "Error"]

    for col, frame in enumerate(selected_frames):
        gt_frame = voltage_gt[frame]
        pred_frame = voltage_pred[frame]
        err_frame = voltage_err[frame]

        # Plot data
        im0 = axs[0, col].imshow(gt_frame, cmap='viridis', vmin=vmin, vmax=vmax)
        axs[0, col].set_title(f"Timestep {frame}", fontsize=10)
        im1 = axs[1, col].imshow(pred_frame, cmap='viridis', vmin=vmin, vmax=vmax)
        im2 = axs[2, col].imshow(err_frame, cmap='bwr', vmin=-err_abs, vmax=err_abs)

        for row in range(3):
            axs[row, col].axis('off')

    # Label rows (leftmost column)
    for row in range(3):
        axs[row, 0].text(
            -0.3, 0.5,               # position relative to image
            #f"{short_labels[row]}\n{row_titles[row]}",
            f"{short_labels[row]}",
            fontsize=12,
            fontweight="bold",
            rotation='vertical',
            ha='center',
            va='center',
            transform=axs[row, 0].transAxes
        )

    # Adjust spacing manually
    plt.subplots_adjust(wspace=0.05, hspace=0.35, top=0.9, bottom=0.05)

    # Add colorbars (shared per row)
    for row, im in enumerate([im0, im1, im2]):
        cbar = fig.colorbar(
            im, ax=axs[row, :].ravel().tolist(),
            fraction=0.02, pad=0.04, orientation='vertical'
        )
        cbar.ax.tick_params(labelsize=8)

    # Title
    fig.suptitle("Selected Voltage Frames: GT, Prediction, and Error", fontsize=16)

    # Save figure
    save_path = os.path.join(save_dir, "Voltage_Samples.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved selected frame visualization to {save_path}")


centre = int(int(args.horizon) / 2)
quarter = int(int(args.horizon) / 4)
three_quarter = int(int(args.horizon) * (3/4))

plot_selected_frames(
    gt=gt,
    pred=pred,
    save_dir=args.data_path,
    selected_frames = [quarter, centre, three_quarter]

)


## ----------------------------------------------------------------------- ##
# ANIMATION OF GT, PREDICTION, AND ERROR (OVER ALL AVAILABLE TIMESTEPS) (with colorbars)
## ----------------------------------------------------------------------- ##
# Get averaged GT and prediction (assuming [B, T, H, W] and using batch 0)

# Use the aligned versions for error calculations
if args.random:
    gt = y_true_aligned
    pred = y_pred_aligned
    voltage_gt = gt[0, :, :, :].detach().cpu().numpy()      # [T, H, W]
    voltage_pred = pred[0, :, :, :].detach().cpu().numpy()  # [T, H, W]

    voltage_err = voltage_gt - voltage_pred                # [T, H, W]
else: 

    voltage_gt = gt[0, :, :, :].detach().cpu().numpy()      # [T, H, W]
    voltage_pred = pred[0, :, :, :].detach().cpu().numpy()  # [T, H, W]
    voltage_err = voltage_gt - voltage_pred                # [T, H, W]

# Consistent color scales
vmin = min(voltage_gt.min(), voltage_pred.min())
vmax = max(voltage_gt.max(), voltage_pred.max())

# Error should be symmetric around 0
err_abs = np.abs(voltage_err).max()
#errmin = max(voltage_err.min(), voltage_err.min())
#errmax = max(voltage_err.max(), voltage_err.max())


fig, axs = plt.subplots(1, 3, figsize=(15, 5))
gt_ax, pred_ax, err_ax = axs

# Initial images
gt_im = gt_ax.imshow(voltage_gt[0], cmap='viridis', vmin=vmin, vmax=vmax)
pred_im = pred_ax.imshow(voltage_pred[0], cmap='viridis', vmin=vmin, vmax=vmax)
err_im = err_ax.imshow(voltage_err[0], cmap='bwr', vmin=-err_abs, vmax=err_abs)

# Titles
gt_ax.set_title("Ground Truth - Voltage")
pred_ax.set_title("Prediction - Voltage")
err_ax.set_title("Error (GT - Pred)")

# Add colorbars
fig.colorbar(gt_im, ax=gt_ax, fraction=0.046, pad=0.04)
fig.colorbar(pred_im, ax=pred_ax, fraction=0.046, pad=0.04)
fig.colorbar(err_im, ax=err_ax, fraction=0.046, pad=0.04)

for ax in axs:
    ax.axis('off')

fig.suptitle("Voltage Channel Evolution Over Time", fontsize=14)

# Update function
def update(frame):
    gt_im.set_array(voltage_gt[frame])
    pred_im.set_array(voltage_pred[frame])
    err_im.set_array(voltage_err[frame])
    fig.suptitle(f"Voltage Channel Evolution Over Time\nTimestep {frame}", fontsize=14)
    return gt_im, pred_im, err_im

# Animation (removed blit=True so all frames refresh)
ani = animation.FuncAnimation(
    fig, update, frames=range(voltage_gt.shape[0]), interval=100, blit=False
)

# Save as GIF
gif_path = os.path.join(args.data_path, "Voltage_Evolution_GT_Pred_Error.gif")
ani.save(gif_path, writer=PillowWriter(fps=10))
plt.close()

'''

plot_selected_frames_with_input(
    x=x_true_full,
    gt=y_true_full,
    pred=y_pred_full,
    save_dir=args.data_path,
    selected_frames=[quarter, centre, three_quarter]
)
## ----------------------------------------------------------------------- ##
# MSE PLOTTING OF V AND W OVER ALL TIMESTEPS 
## ----------------------------------------------------------------------- ##

# Save the mse results for model comparison
mse_data = {
    "model_name": args.data_path,  
    "timesteps": list(range(len(mse_per_timestep_list))),
    "mse": {
        "Voltage": mse_per_timestep_list[0],
        "Recovery": mse_per_timestep_list[1]
    }
}
save_path = os.path.join(args.data_path, "mse_per_timestep.json")
with open(save_path, "w") as f:
    json.dump(mse_data, f, indent=2)


# Save the rmse results for model comparison
rmse_data = {
    "model_name": args.data_path,  
    "timesteps": list(range(len(rmse_per_timestep_list))),
    "rmse": {
        "Voltage": rmse_per_timestep_list[0],
        "Recovery": rmse_per_timestep_list[1]
    }
}
save_path = os.path.join(args.data_path, "rmse_per_timestep.json")
with open(save_path, "w") as f:
    json.dump(rmse_data, f, indent=2)

#print(mse_per_timestep_list)

#mse_per_timestep_array = np.array(mse_per_timestep_list)
mse_per_timestep_array = np.stack(mse_per_timestep_list)    
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
#print(mse_per_timestep_array.shape)
num_timesteps = np.arange(mse_per_timestep_array.shape[1])
for ch in [0, 1]:
    ax = axes[ch]
    ax.plot(num_timesteps, mse_per_timestep_array[ch], color='black', linewidth=2)
    ax.set_title(f'{channel_names[ch]}: Mean Square Error Over Time', fontsize=12)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Mean Squared Error')
    ax.grid(False)
    #ax.set_ylim(0, 0.25)

plt.tight_layout()
save_path = os.path.join(args.data_path, 'MSE_Evolution.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.close(fig)

rmse_per_timestep_array = np.stack(rmse_per_timestep_list)    
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
#print(mse_per_timestep_array.shape)
num_timesteps = np.arange(rmse_per_timestep_array.shape[1])
for ch in [0, 1]:
    ax = axes[ch]
    ax.plot(num_timesteps, mse_per_timestep_array[ch], color='black', linewidth=2)
    ax.set_title(f'{channel_names[ch]}: Root Mean Square Error Over Time', fontsize=12)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Root Mean Squared Error')
    ax.grid(False)
    #ax.set_ylim(0, 0.25)

plt.tight_layout()
save_path = os.path.join(args.data_path, 'RMSE_Evolution.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.close(fig)

'''
