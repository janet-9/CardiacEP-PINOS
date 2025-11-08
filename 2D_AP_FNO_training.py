"""
Training an FNO/PINO model on 2D simulations of Aliev-Panfilov Cell Model 
=============================
Based on the plot_FNO_darcy.py example from the neuralop library. 

Optional Arguments: 
    - Dataset to train on (specify via filepath)
    - Training and testing set size - informed based on the size of the dataset
    - Batch sizes for testing and training 
    - Modes to keep during fourier transform step 
    - Number of hidden channels to use
    - Flag to incorporate physics loss into the model 

"""

#Import required modules for constructing and training the model.
import torch
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# NeuralOps and data loading 
from neuralop.models import FNO, FNO1d, FNO2d, FNO3d 
from neuralop.layers.embeddings import GridEmbeddingND
from AP_neuralop_utils import Trainer
from neuralop.training import AdamW
from AP_neuralop_utils import load_2D_AP
from neuralop.utils import count_model_params
from neuralop.losses import H1Loss, HdivLoss, MSELoss
from neuralop.losses import Aggregator, SoftAdapt, Relobralo
from AP_neuralop_utils import RMSELoss, APLoss, OperatorBackboneLoss, WeightedSumLoss, LpLoss, BoundaryLoss, ICLoss, BCNeumann, APFFTLoss, AdaptiveTrainingLoss
import ast

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
    parser.add_argument('-o', '--output', dest='output', type =str, default = 'training_output_log', help = 'Output file to log the construction information of the input data. Default = %(default)s')
    parser.add_argument('-n', '--n-train', dest='n_train', default = 100, help='Number of training samples. Default is %(default)s, but adapts to dataset info')
    parser.add_argument('-nt', '--n-test', dest='n_test', default = 10, help='Number of testing samples. Default is %(default)s, but adapts to dataset info')
    
    parser.add_argument('-tr', '--train-res', dest='train_res', default = 101, help='Resolution of training data (n x n). Default is n = %(default)s')
    parser.add_argument('-te', '--test-res', dest='test_res', type = list, default = [101], help='List of resolutions of testing data (n x n). Default is [n] = %(default)s')
    parser.add_argument('-s', '--mesh-size', dest='mesh_size', default = 10, help='Size of the n x n mesh in cm. Default is n = %(default)s')
    parser.add_argument('-c', '--conmul', dest='conmul', default = 1.0, help='Conductivity multipler for simulated dataset. Default is n = %(default)s')

    # Arguments for the FNO and training parameters 
    parser.add_argument('-epini', '--epochs_init', dest = 'init_epochs', default = 10, type = int, help = 'Number of epochs to run the initial data driven training model for. Default is %(default)s')
    parser.add_argument('-ep', '--epochs', dest = 'epochs', default = 50, type = int, help = 'Number of epochs to run the training model for. Default is %(default)s')
    parser.add_argument('-ba', '--batch-size', dest='batch_size', default = 15, help='Batch size for training data. Default is %(default)s')
    parser.add_argument('-bt', '--batch-size-test', dest='batch_size_test', default = 5, help='Batch size for testing data. Default is %(default)s')
    parser.add_argument('-m', '--modes', dest='modes', default = 16, help='Number of modes to keep during Fourier transform. Default is %(default)s')
    
    #Channels
    parser.add_argument('-ch', '--channels', dest='ch', default = 2, type = int, help='Number of channels in the dataset. Default is %(default)s - Voltage. Pass 2 for Voltage and Recovery Current')
    parser.add_argument('-hc', '--hidden-channels', dest='hidden_channels', default = 32, help='Number of hidden channels for the FNO. Default is %(default)s')
    
    #Physics Loss Parameters 
    parser.add_argument('-phys', '--phys-loss', dest = 'phys_loss', action = "store_true", help = 'Toggle to train using the physics loss. Default is data only')
    parser.add_argument('-p_meth', '--phys-meth', dest = 'phys_method', type = str, choices= {'finite_difference', 'finite_difference_fft', 'query_point'},  default = 'finite_difference', help = 'Method for calculating the physics loss. Default is %(default)s')
    parser.add_argument('-adapt', '--adapt', dest = 'adapt', action = "store_true", help = 'Option to apply weight adaptation in secondary training round.')
    
    
    parser.add_argument('-vl', '--vloss', dest='v_loss', default = 1.0, type = float, help='Weighting of the voltage pde loss for PINO loss function. Default is %(default)s')
    parser.add_argument('-wl', '--wloss', dest='w_loss', default = 0.0, type = float, help='Weighting of the recovery current pde loss for PINO loss function. Default is %(default)s')
    parser.add_argument('-ic', '--icloss', dest='ic_loss', default = 0.1, type = float, help='Weighting of the initial conditions loss for PINO loss function. Default is %(default)s')
    parser.add_argument('-bc', '--bcloss', dest='bc_loss', default = 0.1, type = float, help='Weighting of the boundary conditions loss for PINO loss function. Default is %(default)s')
    parser.add_argument('-res', '--resloss', dest='res_loss', default = 0.01, type = float, help='Weighting of the residual PDE loss for PINO loss function. Default is %(default)s')
    parser.add_argument('-bound', '--boundary', dest='boundary', default = 0.1, type = float, help='Weighting for the GT-Pred boundary condition loss. Defaut is %(default)s')
    parser.add_argument('-data', '--data', dest='data_loss', default = 1.0, type = float, help='Weighting for the GT data in PINO training. Defaut is %(default)s')

    parser.add_argument('-D', '--D', dest='D', default = 0.55 , type = float, help='Value for parameter D in AP model. Default is %(default)s')
    parser.add_argument('-K', '--K', dest='K', default = 8.0 , type = float, help='Value for parameter K in AP model. Default is %(default)s')
    parser.add_argument('-a', '--a', dest='a', default = 0.15 , type = float,help='Value for parameter a in AP model. Default is %(default)s')
    parser.add_argument('-b', '--b', dest='b', default = 0.15 , type = float,help='Value for parameter b in AP model. Default is %(default)s')
    parser.add_argument('-e', '--epsilon', dest='epsilon', type = float,default = 0.002 , help='Value for parameter epsilon in AP model. Default is %(default)s')
    parser.add_argument('-mu1', '--mu1', dest='mu1', default = 0.2 , type = float,help='Value for parameter mu1 in AP model. Default is %(default)s')
    parser.add_argument('-mu2', '--mu2', dest='mu2', default = 0.3 , type = float,help='Value for parameter mu2 in AP model. Default is %(default)s')
    parser.add_argument('-ts', '--t_scale', dest='t_scale', default = 12.9 , type = float, help='Value for scaling time to AU. Default is %(default)s')

    #Model saving and storage
    parser.add_argument('-em', '--eval-metric', dest='eval_metric', default = 'mse', help='Evaluation Metric for saving the best model. Default is %(default)s')
    parser.add_argument('-r', '--results', dest='results', default = 'Results', help='File path for storing training and evaluation results. Default is %(default)s')
    parser.add_argument('-eres', '--eval-res', dest='eval_res', default = 101, help='Resolution of training data (n x n) to use for the evaluation metric. Default is n = %(default)s')
    parser.add_argument('-econ', '--eval-con', dest='eval_con', default = 1.0, help='Conductivity multipler to use for the evaluation metric. Default is n = %(default)s')
    parser.add_argument('-rnd', '--random-split', dest='random_split', action='store_true', help='Random test-train split used in traning')
    parser.add_argument('-frame', '--frames', dest='frames', type = int, default = 1, help='Number of frames used in the input-output training. Default = %(default)s')
    args = parser.parse_args()


## ----------------------------------------------------------------------- ##
# FILE SET FOR SAVING RESULTS
## ----------------------------------------------------------------------- ##

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


# Determine if dataset is planar, centrifugal, spiral or spiral-breakup
if "stable" in dataset_path_lower:
    dataset_type = "stable"
elif "chaotic" in dataset_path_lower:
    dataset_type = "chaotic"
elif "planar" in dataset_path_lower:
    dataset_type = "planar"
elif "centrifugal" in dataset_path_lower:
    dataset_type = "centrifugal"
else:
    dataset_type = "Unknown"

# Build results folder name based on physics loss and dataset type
if args.random_split:
    rnd = 1
else:
    rnd = 0

if args.phys_loss: 
    results_folder_name = (f"{args.results}_{dataset_type}_PINO_"
                           f"{args.phys_method}_{args.res_loss}_resloss_{args.ic_loss}_icloss_{args.bc_loss}_bcloss_{args.w_loss}_wloss_{args.eval_metric}_eval_{args.D}_D_{args.frames}_frames_{timestamp}")
else: 
    results_folder_name = f"{args.results}_{dataset_type}_Data_Only_{args.eval_metric}_{args.frames}_frames_{timestamp}"


results_path = os.path.join(results_folder_name)

# Create the directory if it doesn't exist
os.makedirs(results_path, exist_ok=True)
print(f"Saving results to: {results_path}")


# Redirect stdout to save as a log file as well as being printed to the consol
dataset_info = f'{args.output}.txt'
log_file_path = os.path.join(results_path, dataset_info)
# Ensure directory exists
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

log_file = open(log_file_path, 'w')
sys.stdout = Tee(sys.__stdout__, log_file)


#Set the device for training:
device = 'cuda'

'''
wandb.init(
    entity= 'hannah-f-lydon-king-s-college-london',
    project= f"NO-tests-2DAPChaotic-vanFNO",   
    name= args.data_path,         
)
'''

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
time_boundary = (float(delta_t) * (float(time_frames) -1) )

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
print(f"Mesh size: {args.mesh_size} cm x {args.mesh_size} cm")
#if time_frames > 1:
time_boundary = (float(delta_t) * (float(time_frames) -1) )
#else:
    #time_boundary = float(delta_t)
print(f"Sample Time Boundary: {time_frames} frames with spacing {delta_t} ms =  {time_boundary} ms ")
embedding = GridEmbeddingND(in_channels=args.ch, dim=3, grid_boundaries=[[0,float(time_boundary)], [0, args.mesh_size], [0, args.mesh_size]])
print("Embedding Grid Boundaries =", embedding.grid_boundaries)

'''
# Establish the embedding and boundaries for the spatial dimensions only (Adapting the spatial embeddings for the grid resolutions):
print('\n### --------- ###\n')
print(f"Mesh size: {args.mesh_size} cm x {args.mesh_size} cm")
embedding = GridEmbeddingND(in_channels=args.ch, dim=2, grid_boundaries=[[0, args.mesh_size], [0, args.mesh_size]])
print("Embedding Grid Boundaries =", embedding.grid_boundaries)
'''

# Load one training sample for inspection
train_dataset = train_loader.dataset
index = 0
data = train_dataset[index]
data = data_processor.preprocess(data, batched=True)  # shape: [channels, time, H, W]
channels, time_steps, H, W = data['x'].shape
print(f"Sample Data shape: {data['x'].shape}: channels={channels}, time={time_steps}, H={H}, W={W}\n")
print('\n### --------- ###\n')

## ----------------------------------------------------------------------- ##
# CREATING THE MODEL
## ----------------------------------------------------------------------- ##


# Building the FNO model structure for training 
model = FNO(n_modes=(8, 16, 16),
             in_channels=args.ch, 
             out_channels=args.ch,
             hidden_channels=32, 
             projection_channel_ratio=2,
             positional_embedding = embedding)
model = model.to(device)
n_params = count_model_params(model)
print(f'\nModel has {n_params} parameters.')
#sys.stdout.Tee()

## TRAINING SETUP ##

# Define optimisation 
optimizer = AdamW(model.parameters(), 
                                lr=5e-4, 
                                weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
#scheduler =torch.optim.lr_scheduler.ReduceLROnPlateau(...)

## ----------------------------------------------------------------------- ##
# DEFINING THE LOSSES   
## ----------------------------------------------------------------------- ##

# data losses 
l2loss = LpLoss(d=2, p=2)
mse = MSELoss()
rmse = RMSELoss()
boundary = BoundaryLoss()
ic = ICLoss()
bcn = BCNeumann()

#conductivity scaling: 
D = args.D * float(args.conmul)

# FFT FDM loss 
apfft = APFFTLoss(k=args.K, a=args.a, epsilon=args.epsilon, mu1=args.mu1, mu2=args.mu2, b=args.b, D=args.D, Lt=float(time_boundary), t_scale=args.t_scale)

# physics losses
apfdm = APLoss( k=args.K, 
                 a=args.a, 
                 epsilon=args.epsilon, 
                 mu1=args.mu1, 
                 mu2=args.mu2, 
                 b=args.b,
                 D = D,
                 method = args.phys_method, 
                 delta_t = delta_t, 
                 Lt = time_boundary,
                 dx = dx, 
                 dy = dy, 
                 v_loss_weighting = args.v_loss, 
                 w_loss_weighting = args.w_loss,
                 ic_weighting=args.ic_loss,
                 bc_weighting=args.bc_loss,
                 res_weighting=args.res_loss,
                 t_scale = args.t_scale, 
                 V_rest = V_rest,
                 V_amp = V_amp
)

# Determine which losses to use for training based on physics loss weight
# resloss is the physics-based loss (either FDM or FFT)

if args.phys_method == "finite_difference":
    resloss = apfdm
    print("Calculating Residual Loss via Finite Difference")
elif args.phys_method == "finite_difference_fft":
    resloss = apfft
    print("Calculating Residual Loss via FFT Finite Difference")
else:
     print("Unknown method for residual loss calculation!")


# Establish Global Evaluation losses:
eval_losses = {
        'l2': l2loss,
        'mse': mse,
        'rmse': rmse,
        'ap_phys': resloss,
        'boundary': boundary,
        'ic': ic,
        'bcn': bcn
    }

## ----------------------------------------------------------------------- ##
# INITIAL TRAINING ROUND: DATA ONLY LOSS
## ----------------------------------------------------------------------- ##
train_loss = l2loss

# Training the model
# ---------------------
# Print out the training info 
print('\n### INITIAL TRAINING ROUND ###\n')
print('\n### MODEL ###\n', model)
print('\n### OPTIMIZER ###\n', optimizer)
print('\n### SCHEDULER ###\n', scheduler)
print('\n### LOSSES ###')
print(f'\n * Train: {train_loss}')
print(f'\n * Test: {eval_losses}')
print(f'\n * Initial Training Round for {args.init_epochs} epochs')
#sys.stdout.flush()

# Create the trainer:
trainer = Trainer(model=model, n_epochs=args.init_epochs,
                  device=device,
                  data_processor=data_processor,
                  wandb_log=True,
                  eval_interval=10,
                  use_distributed=False,
                  verbose=True
                  )

#Train the model on the loaded dataset for a data pnly FNO training run
eval_metric = f"({args.eval_res}, {args.eval_con})_{args.eval_metric}"
print(f"Saving best model according to {eval_metric}")
json_log_path=os.path.join(results_path, "training_log.json")
trainer.train(train_loader=train_loader,
              test_loaders=test_loaders,
              optimizer=optimizer,
              scheduler=scheduler, 
              regularizer=False, 
              training_loss=train_loss,
              eval_losses=eval_losses,
              save_best = eval_metric,
              save_dir = results_path,
              json_log_path = json_log_path
              )

# Convert the json training log file into a valid file to read in the results analysis:
training_log_json = os.path.join(results_path, "training_log.json")
with open(training_log_json, 'r') as file:
    data = [json.loads(line) for line in file if line.strip()]

with open(training_log_json, 'w') as file:
    json.dump(data, file, indent=2)

sys.stdout = sys.__stdout__
log_file.close()
#wandb.finish()
