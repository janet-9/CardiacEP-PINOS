# CardiacEP-PINOS

Repository for the work **"PHYSICS INFORMED NEURAL OPERATORS FOR CARDIAC ELECTROPHYSIOLOGY"**.  

This repository provides training and evaluation scripts for running a Physics-Informed Neural Operator (PINO) based model for the simulation of cardiac electrophysiology propagation scenarios using the Aliev Panfilov (AP) cardiac cell model. We also provide the results of the experiments performed in this work, including animations of the model predictions. 

## Repository Overview

**Data**
Contains the scripts for transforming simulation data into training, testing and evalaution datasets and an example dataset folder for the planar, centrifugal, spiral and spiral-break propagation scenarios. 

**Paper_Results**
Evaluation results for the experiments detailed in the accompanying work "PHYSICS INFORMED NEURAL OPERATORS FOR CARDIAC ELECTROPHYSIOLOGY", including the baseline tests, mesh resolution testing, and zero shot transfer experiments. All of the results folders contain side by side animations of the predicted and ground truth simulations. 

**Scripts**
Training script for the PINO model, and evalution scripts for assessing the model on a 'Point to Point' or 'Roll-Out' basis, as discussed in the paper. Also contains a script for comparing the results of different models on the same evaluation dataset. 

**Usage**
Install the neuralops library from: https://github.com/neuraloperator/neuraloperator following the installation instructions. 

Run the training and evaluation scripts found in scripts using relevant arguments for your experiments. 





