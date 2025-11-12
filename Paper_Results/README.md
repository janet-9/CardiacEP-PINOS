# Paper Results
---

### **Baseline_Results**

- 1. **Baseline_results_200_frame**  
       Results for each of the trained PINO models for each considered propagation scenarios, grouped by the type of frame training and the evaluation method. All evaluation trajectories were fixed at 1000ms. 

- 2. **Baseline_results_500_frame**  
       Results for each of the trained PINO models for stable and chaotic propagation scenarios, grouped by the type of frame training and the evaluation method. All evaluation trajectories were fixed at 2500ms. 

<p align="center">
<img src="Baseline_Results_500_frame/stable/5_frame_Rollout/Results_stable_PINO_2.0_finite_difference_0.01_resloss_0.1_icloss_0.1_bcloss_0.0_wloss_mse_eval_0.55_D_5_frames_31_10_2025-09_45/Voltage_Evolution_GT_Pred_Error.gif" width="80%" alt="Side by side animation of ground truth and predicted model for multi-frame rollout simulation of the stable spiral propagation scenario">
</p>

---
### **Epoch Results**

- 1. Results for each of the trained PINO models evaluated on trajectories of 1000ms for planar and centrifugal propagation scenarios, and 2500ms for the stable and chaotic scenarios. 

- 2. Plots demonstrating the change in evaluation RMSE values for each training run. 
<p align="center">
  <img src="Epoch_Results/Epoch_comparison_1000ms_NORM.png" width="80%" alt="Performance evaluation via trainng epochs">
</p>

--
### **Mesh Resolution Results (x10)**

- 1. Results for each of the PINO models for each scenario trained on a low (downsampled) resolution and evaluated on the groudn truth resolution. (trained on 41x41 and evaluated on 401x401). 

- 2. Plots demonstrating the change in evaluation RMSE values for each evaluation resolution
<p align="center">
  <img src="Mesh_Resolution_Results_x10/Resolution_Invariance.png" width="60%" alt="Performance evaluation for different training resolutions">
</p>

--
### **Zero Shot Transfer**

- 1. **Zero_shot_centri2stable** : Compares the predictions on the stable propagation set using the model trained on stable propagation data and on centrifugal propagation data. 

- 2. **Zero_shot_planar2stable** : Compares the predictions on the stable propagation set using the model trained on stable propagation data and on planar propagation data. 
