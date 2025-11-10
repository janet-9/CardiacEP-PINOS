"""
Inspecting the tensor data to use in PINO/FNO training
=============================
"""

#Import required modules for constructing and training the model.
import torch
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import ast

# NeuralOps and data loading 
from neuralop.models import FNO, FNO1d, FNO2d, FNO3d 
from neuralop.layers.embeddings import GridEmbeddingND
from AP_neuralop_utils import Trainer
from neuralop.training import AdamW
from AP_neuralop_utils import load_2D_AP
from neuralop.utils import count_model_params
from neuralop.losses import LpLoss, H1Loss, HdivLoss, MSELoss
from AP_neuralop_utils import RMSELoss, APLoss, OperatorBackboneLoss, WeightedSumLoss
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter 

# Device and system imports 
import sys
import random
import argparse
import os
from pathlib import Path
import re

#imports required for data logging
import json
import wandb
from datetime import datetime 


# Define the optional arguments to use when calling the operator
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Arguments for the data structure:
    parser.add_argument('-d', '--data-path', dest='data_path', required = True, type = str, help='Input data file path to train and test on')
    parser.add_argument('-de', '--data-path-eval', dest='data_path_eval', required = True, type = str, help = 'File path for unseen dataset to evaluate on')
    parser.add_argument('-o', '--output', dest='output', type =str, default = 'data_inspection', help = 'Output file to log the inspection information of the input data. Default = %(default)s')
    parser.add_argument('-n', '--n-train', dest='n_train', default = 100, help='Number of training samples. Default is %(default)s, but adapts to dataset info')
    parser.add_argument('-nt', '--n-test', dest='n_test', default = 10, help='Number of testing samples. Default is %(default)s, but adapts to dataset info')
    
    parser.add_argument('-tr', '--train-res', dest='train_res', default = 26, help='Resolution of training data (n x n). Default is n = %(default)s')
    parser.add_argument('-te', '--test-res', dest='test_res', type = list, default = [101], help='List of resolutions of testing data (n x n). Default is [n] = %(default)s')
    parser.add_argument('-s', '--mesh-size', dest='mesh_size', default = 10, help='Size of the n x n mesh in cm. Default is n = %(default)s')
    parser.add_argument('-c', '--conmul', dest='conmul', default = 1.0, help='Conductivity multipler for simulated dataset. Default is n = %(default)s')

    # Arguments for the FNO and training parameters 
    parser.add_argument('-ba', '--batch-size', dest='batch_size', default = 15, help='Batch size for training data. Default is %(default)s')
    parser.add_argument('-bt', '--batch-size-test', dest='batch_size_test', default = 5, help='Batch size for testing data. Default is %(default)s')
    parser.add_argument('-m', '--modes', dest='modes', default = 16, help='Number of modes to keep during Fourier transform. Default is %(default)s')
    parser.add_argument('-dt', '--delta_t', dest = 'delta_t', type = float, default = 1.0, help = 'Timestep between frames for simulation data, Default is %(default)s')
    #Channels
    parser.add_argument('-ch', '--channels', dest='ch', default = 2, type = int, help='Number of channels in the dataset. Default is %(default)s - Voltage. Pass 2 for Voltage and Recovery Current')
    parser.add_argument('-hc', '--hidden-channels', dest='hidden_channels', default = 32, help='Number of hidden channels for the FNO. Default is %(default)s')

    args = parser.parse_args()


## ----------------------------------------------------------------------- ##
# Define the query point sampling
## ----------------------------------------------------------------------- ##

def sample_query_points(B, H, W, n_interior=5000, n_boundary=400, 
                            Lx=10.0, Ly=10.0, device = 'cpu'):
        """
        Sample interior and boundary query points for autograd-based losses.

        Parameters
        ----------
        B : int
            Batch size
        H, W : int
            Grid resolution
        n_interior : int
            Number of interior query points per batch element
        n_boundary : int
            Number of query points per boundary (left/right/top/bottom)
        Lx, Ly : float
            Physical domain dimensions (cm)
        device : str
            Torch device

        Returns
        -------
        query_dict : dict
            {
            'interior': torch.Tensor [B, n_interior, 2] (normalized [-1,1]),
            'boundaries': dict with 'left','right','top','bottom' query points
            }
        """
        # ----- Interior points -----
        x_int = torch.rand(B, n_interior, 1, device=device) * Lx
        y_int = torch.rand(B, n_interior, 1, device=device) * Ly
        interior_xy = torch.cat([x_int, y_int], dim=-1)

        # normalize to [-1, 1]
        interior_norm = interior_xy.clone()
        interior_norm[..., 0] = 2.0 * interior_xy[..., 0] / Lx - 1.0
        interior_norm[..., 1] = 2.0 * interior_xy[..., 1] / Ly - 1.0

        # ----- Boundary points -----
        y_b = torch.rand(B, n_boundary, 1, device=device) * Ly
        x_b = torch.rand(B, n_boundary, 1, device=device) * Lx

        boundaries = {
            "left":  torch.cat([torch.zeros_like(y_b), y_b], dim=-1),
            "right": torch.cat([torch.full_like(y_b, Lx), y_b], dim=-1),
            "top":   torch.cat([x_b, torch.zeros_like(x_b)], dim=-1),
            "bottom":torch.cat([x_b, torch.full_like(x_b, Ly)], dim=-1),
        }

        # normalize boundaries
        boundaries_norm = {}
        for key, xy in boundaries.items():
            xy_norm = xy.clone()
            xy_norm[..., 0] = 2.0 * xy[..., 0] / Lx - 1.0
            xy_norm[..., 1] = 2.0 * xy[..., 1] / Ly - 1.0
            boundaries_norm[key] = xy_norm

        return {"interior": interior_norm, "boundaries": boundaries_norm}

# Define a class for logging the outputs of the training process
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() 
    def flush(self):
        for f in self.files:
            f.flush()

# Generate the results folder to store the training logs and results 
timestamp = datetime.now().strftime("%d_%m_%Y-%H_%M")
dataset_path_lower = args.data_path.lower()


# Determine if dataset is stable or chaotic
if "stable" in dataset_path_lower:
    dataset_type = "Stable"
elif "chaotic" in dataset_path_lower:
    dataset_type = "Chaotic"
elif "centri" in dataset_path_lower:
    dataset_type = "Centrifugal"
elif "planar" in dataset_path_lower:
    dataset_type = "Planar"
else:
    dataset_type = "Unknown"

# 
print(f"Saving results to: {args.data_path}")


# Redirect stdout to save as a log file as well as being printed to the consol
dataset_info = f'{args.output}.txt'
log_file_path = os.path.join(args.data_path, dataset_info)
# Ensure directory exists
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

log_file = open(log_file_path, 'w')
sys.stdout = Tee(sys.__stdout__, log_file)

#Set the device for training:
device = 'cpu'

## ----------------------------------------------------------------------- ##
#  LOADING AND PREPARING THE DATASETS (WITH INSPECTIONS)
## ----------------------------------------------------------------------- ##


# Define the path to find the dataset to load from 
def get_data_root(custom_path: str):
    return (Path.cwd() / custom_path).resolve()

print('\n### --------- ###\n')
custom_path = args.data_path
example_data_root = get_data_root(custom_path)
print(f"Loading datasets from {example_data_root}")

# Define the size of the training and testing set sizes using the dataset_info file in the folder (if available):

#Default values for training (if no dataset info file is available)
n_train = args.n_train
train_res = args.train_res

#Use the dataset info to extract the relevant parameters for embedding and training set up 
target_name = f"dataset_info_{args.train_res}_{args.conmul}.txt"
dataset_info = os.path.join(args.data_path, target_name)
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


# Default arguments for testing data:
n_test = args.n_test
test_res = args.test_res

# Extract testing data information: 
testing_dataset_info_files = [
    os.path.join(args.data_path, f)
    for f in os.listdir(args.data_path)
    if f.startswith("dataset_info_") and f.endswith(".txt")
]

#print(f"LOADING TEST DATASETS: {testing_dataset_info_files}")


test_resolutions = []   # store all test resolutions here
cm_tests = [] # store all conductivity multiplier here
n_tests = [] # store number of testing samples here
batch_tests = [] #store the size of the testng batches here

for dataset_info in testing_dataset_info_files:
    basename = os.path.basename(dataset_info)
    match = re.match(r"dataset_info_(\d+)_([\d\.]+)\.txt", basename)
    if not match:
        print(f"Skipping file with unexpected name format: {basename}")
        continue
    res_str, conmul_str = match.groups()
    res = int(res_str)
    conmul = float(conmul_str) if '.' in conmul_str else int(conmul_str)
    with open(dataset_info, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('Testing data shapes:'):
                numbers = re.findall(r'\d+', line)
                if numbers:
                    n_test = int(numbers[0])
                    #print(f"n_test: {n_test}")
                    n_tests.append(n_test)
                    res = int(numbers[3]) 
                    test_resolutions.append(res)
                    batch_tests.append(int(args.batch_size_test))
    cm_tests.append(conmul)

print('\n### --------- ###\n')
print(f" Number of testing samples: {n_tests}")
print(f" Testing resolutions: {test_resolutions}") 
print(f" Testing Conductivities: {cm_tests}")
print(f" Testing batch sizes: {batch_tests}")
print('\n### --------- ###\n')

# Load the datasets (using the new loading function defined for the dataset): 
train_loader, test_loaders, data_processor = load_2D_AP(
        n_train=n_train, batch_size=int(args.batch_size),
        train_resolution= train_res,
        test_resolutions= test_resolutions, n_tests=n_tests,
        test_batch_sizes= batch_tests, data_root=example_data_root, dataset_name = '2D_AP',
        cm_train = args.conmul, cm_tests = cm_tests,
        encode_input = False, encode_output = False,
)
data_processor = data_processor.to(device)


# Establish the embedding and boundaries for the inclusion of the time channel (Adapting the spatial embeddings for the grid resolutions):
print('\n### --------- ###\n')
print(f" Mesh size: {args.mesh_size} cm x {args.mesh_size} cm")
time_boundary = float(delta_t) * float(time_frames)
print(f" Sample Time Boundary: {delta_t} ms * {time_frames} frames =  {time_boundary} ms ")
embedding = GridEmbeddingND(in_channels=args.ch, dim=3, grid_boundaries=[[0,float(time_boundary)], [0, args.mesh_size], [0, args.mesh_size]])
print("grid_boundaries =", embedding.grid_boundaries)

# Load one training sample for inspection
train_dataset = train_loader.dataset
test_loader = next(iter(test_loaders.values()))  # get first DataLoader
test_dataset = test_loader.dataset

index = 0
data = train_dataset[index]
data = data_processor.preprocess(data, batched=True)  # shape: [channels, time, H, W]

print(f"Data shape: {data['x'].shape}\n")

# Ensure batch dimension exists
if data['x'].ndim == 4:
    data['x'] = data['x'].unsqueeze(0)  
if data['y'].ndim == 4:
    data['y'] = data['y'].unsqueeze(0)  

batch, channels, time_steps, H, W = data['x'].shape
print(f"Data shape: batch={batch}, channels={channels}, time={time_steps}, H={H}, W={W}\n")

# Sampling the query points to use in the physics loss
n_interior = 1500
n_boundary = 50
Lx, Ly = args.mesh_size, args.mesh_size
query_points = sample_query_points(
            batch, H, W, n_interior=n_interior, n_boundary=n_boundary,
            Lx=Lx, Ly=Ly
        )
print('\n### --------- ###\n')
print('-- Inspecting Query Points --')

#print(query_points)

# Plot the query points onto the mesh 
# Pick sample slices: first batch, first channel:
# Input: Last Frame 
# Output: First Frame

sample_slice_input = data['x'][0, 0, 4]  # shape: [H, W]
print(sample_slice_input.shape)
sample_slice_output = data['y'][0, 0, 0]  # shape: [H, W]
print(sample_slice_output.shape)

# Get grid resolution
H, W = sample_slice_output.shape

# Extract query points from dictionary (first batch element)
interior_points = query_points['interior'][0]       # [n_interior, 2]
boundaries = query_points['boundaries']            # dict with 'left', 'right', 'top', 'bottom'

# Convert normalized coordinates [-1,1] to pixel indices
def norm_to_pixel(norm_xy, H, W):
    """
    Convert normalized coordinates ([-1,1]) to pixel coordinates ([0,H) & [0,W)).
    norm_xy: Tensor [..., 2]
    Returns: numpy array of shape [..., 2] with (x_pixel, y_pixel)
    """
    x_pix = ((norm_xy[..., 0] + 1) / 2) * (W - 1)
    y_pix = ((norm_xy[..., 1] + 1) / 2) * (H - 1)
    return torch.stack([x_pix, y_pix], dim=-1).cpu().numpy()

#Interior points
interior_pix = norm_to_pixel(interior_points, H, W)
boundary_pix = {k: norm_to_pixel(v[0], H, W) for k, v in boundaries.items()}
# Boundary points (Combine all boundary points into one big array)
all_boundary_pts = np.concatenate(list(boundary_pix.values()), axis=0)

# Plot the input-output samples with query points 
plt.figure(figsize=(6, 6))
fig, (ax1, ax2) = plt.subplots(1,2)
#fig.suptitle(f"Sample Slices with Query Points: Sample {index}")

#Input Slice 
ax1.imshow(sample_slice_input.cpu().numpy(), origin='lower', cmap='viridis')
# Interior amd boundary points
ax1.scatter(interior_pix[:, 0], interior_pix[:, 1], 
            s=2, c='red', marker = 'x', alpha=0.4, label='Interior')
ax1.scatter(all_boundary_pts[:, 0], all_boundary_pts[:, 1], 
            s=2, marker='x', c='cyan', label='Boundary')
ax1.set_xlabel("X (pixel)")
ax1.set_ylabel("Y (pixel)")
ax1.set_title("Final Input Frame")

#Output Slice 
ax2.imshow(sample_slice_output.cpu().numpy(), origin='lower', cmap='viridis')
# Interior amd boundary points
ax2.scatter(interior_pix[:, 0], interior_pix[:, 1], 
            s=2, c='red', marker = 'x', alpha=0.4, label='Interior')
ax2.scatter(all_boundary_pts[:, 0], all_boundary_pts[:, 1], 
            s=2, marker='x', c='cyan', label='Boundary')
ax2.set_xlabel("X (pixel)")
ax2.set_ylabel("Y (pixel)")
ax2.set_title("First Output Frame")

#plt.legend()
plt.tight_layout()
plot_name = f'Query_point_visualiation_{train_res}.png'
save_path = os.path.join(args.data_path, plot_name)
print(f'Saving query point visualisation to {save_path}')
plt.savefig(save_path, bbox_inches='tight', dpi=300)

print('\n### --------- ###\n')

# Inspect the channel amplitudes and embeddings
batch_idx = 0  # select first batch
for channel_to_inspect in range(data['x'].shape[1]):  # channels
    x_channel = data['x'][batch_idx, channel_to_inspect] 
    print(f"\n--- Inspecting channel {channel_to_inspect} ---")
    print(f"Raw amplitude: min={x_channel.min().item():.4f}, max={x_channel.max().item():.4f}")

    # Add batch+channel dims for embedding
    x_channel_input = x_channel.unsqueeze(0).unsqueeze(0)# [1,1,time,H,W]
    x_emb = embedding(x_channel_input)[0]  # remove batch

    print("Embedding shape:", x_emb.shape)
print('\n### --------- ###\n')


## ----------------------------------------------------------------------- ##
# ANIMATION OF THE INSPECTED TENSOR DATASET
## ----------------------------------------------------------------------- ##
# Get averaged GT and prediction (assuming [B, T, H, W] and using batch 0)

folder_path = args.data_path_eval
target_dataset = f"2D_AP_eval_full_51_1.0.pt"

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename == target_dataset:
        file_path = os.path.join(folder_path, filename)
        tensor = torch.load(file_path, map_location="cuda")
        print('--------------------------------------------------')
        print(f"LOADING EVALUATION DATASET: {filename}")
        print('--------------------------------------------------')

y_true_full = tensor[0, :, :, :].detach().cpu().numpy()  
print(f"Ground truth shape: {y_true_full.shape}")

#train_dataset = train_dataset[0, :, :, :].detach().cpu().numpy()      # [T, H, W]
#test_dataset = test_datset[0, :, :, :].detach().cpu().numpy()  # [T, H, W]

# Consistent color scales
#vmin = min(train_datset.min(), test_datset.min())
#vmax = max(train_datset.max(), test_datset.max())

vmin = min(y_true_full.min(), y_true_full.min())
vmax = max(y_true_full.max(), y_true_full.max())
# Error should be symmetric around 0
#err_abs = np.abs(voltage_err).max()
#errmin = max(voltage_err.min(), voltage_err.min())
#errmax = max(voltage_err.max(), voltage_err.max())


fig, axs = plt.subplots(1, 3, figsize=(15, 5))
gt_ax, pred_ax, err_ax = axs

# Initial images
gt_im = gt_ax.imshow(y_true_full[0], cmap='viridis', vmin=vmin, vmax=vmax)
#pred_im = pred_ax.imshow(test_datset[0], cmap='viridis', vmin=vmin, vmax=vmax)

# Titles
gt_ax.set_title("Ground Truth")
#pred_ax.set_title("Prediction - Train Set")


# Add colorbars
fig.colorbar(gt_im, ax=gt_ax, fraction=0.046, pad=0.04)
#fig.colorbar(pred_im, ax=pred_ax, fraction=0.046, pad=0.04)

for ax in axs:
    ax.axis('off')

fig.suptitle("Voltage Channel Evolution Over Time", fontsize=14)

# Update function
def update(frame):
    gt_im.set_array(y_true_full[frame])
    #pred_im.set_array(test_datset[frame])
    fig.suptitle(f"Voltage Channel Evolution Over Time\nTimestep {frame}", fontsize=14)
    return gt_im

# Animation (removed blit=True so all frames refresh)
ani = animation.FuncAnimation(
    fig, update, frames=range(y_true_full.shape[0]), interval=100, blit=False
)

# Save as GIF
gif_path = os.path.join(args.data_path, "Voltage_Evolution_GT.gif")
ani.save(gif_path, writer=PillowWriter(fps=10))
plt.close()
