import torch
import torch.nn.functional as F
from torch.autograd import grad
from .differentiation import FiniteDiff
import numpy as np
import torch.nn as nn

## NEW: Class for calculating the physics loss for the Aliev-Panfilov Cell Model (Including the ICs and BCs) ##
class APLoss(object):

    '''
    NEW: Introduction of a class to calculate the loss on the ALiev-Panfilov cell model PDE loss. 
    Inspired by the EP-PINNs structuring from the Imperial group and using some inbuilt functions from the neuralops library. 

    This returns a loss that is a weighted combintation of:
    - The residual loss of the PDE, the aim is to get this as close to zero as possible. 
    - The boundary condition loss (EITHER enforcing no-flux conditions OR minimising difference with the GT data). 
    - The initial conditions loss, which aims to match the first sample in a batch to the ground truth value. 
        (EITHER minmising the difference between the last input frame and first output frame OR the first GT output frame) 

    Parameters: 
    -----------
    AP model parameters: Floats
        (k, a, eps, mu1, mu2, b) 
        D = diffusion tensor
    
    loss_fn: Callable, optional
       class used to define the loss. Default is F.mse_loss (mean squared error)

    Coordinate parameters: 
    delta_t = time resolution of the input data 
    dx = spatial resolution of the mesh in x
    dy = spatial resolution of the mesh in y 

    '''

    # Defining the AP model parameters:
    def __init__(self, 
                 D=1.0, k=8.0, a=0.15, epsilon=0.002, mu1=0.2, mu2=0.3, b=0.15,
                 loss_fn=F.mse_loss,
                 method="finite_difference",
                 
                 y_bc_query = 50,
                 
                 delta_t=1.0,
                 dx=1.0,
                 dy=1.0,
                 Lx = 10.0,
                 Ly = 10.0,
                 Lt = 49,
                 t_scale = 12.9,
                 V_rest = -80,
                 V_amp = 100,
                 
                 v_loss_weighting=1.0, 
                 w_loss_weighting=1.0,
                 ic_weighting=1.0,
                 bc_weighting=1.0,
                 res_weighting=1.0,

                 reduction= "sum"
                 ):
        super().__init__()
    
        ## Assign the parameters ##

        #PDE parameters
        self.D = D
        self.k = k
        self.a = a
        self.epsilon = epsilon
        self.mu1 = mu1
        self.mu2 = mu2
        self.b = b
        self.t_scale = t_scale
        self.V_rest = V_rest
        self.V_amp = V_amp 
        
        # Loss calculation parameters
        self.loss_fn = loss_fn
        self.method = method
        # residual weightings
        self.v_loss_weighting = v_loss_weighting
        self.w_loss_weighting = w_loss_weighting 
        # full physics loss weightings
        self.ic_weighting=ic_weighting
        self.bc_weighting=bc_weighting
        self.res_weighting=res_weighting
 
        # Coordinate calculation parameters 
        self.delta_t = delta_t
        self.dx = dx
        self.dy = dy
        

        self.Lx = Lx
        self.Ly = Ly
        self.Lt = Lt
        self.y_bc_query = y_bc_query

        # Reducing losses across the batch 
        self.reduction = reduction

    ## ----------------------------------------------------------------------- ##
    # Laplacian Calculation for the PDE
    ## ----------------------------------------------------------------------- ##

    def laplacian_2d(self, u):
            """
            Compute the 2D Laplacian using central finite differences and Neumann (zero-gradient) boundary conditions.
            
            Args:
                u: Tensor of shape (B, 1, H, W) — batch of scalar fields
                dx: Grid spacing in x-direction
                dy: Grid spacing in y-direction
                
            Returns:
                Tensor of shape (B, 1, H, W) — the Laplacian of u
        
            """

            B, C, H, W = u.shape

            # Compute grid spacings
            dx = self.Lx / (W - 1)
            dy = self.Ly / (H - 1)
            # Define finite difference kernel for 2D Laplacian (5-point stencil)
            kernel = torch.tensor([[0,       1/dy**2,  0],
                                [1/dx**2, -2*(1/dx**2 + 1/dy**2), 1/dx**2],
                                [0,       1/dy**2,  0]], dtype=u.dtype, device=u.device)

            # Reshape kernel to match conv2d input: (out_channels, in_channels, kernel_height, kernel_width)
            kernel = kernel.view(1, 1, 3, 3)

            # Neumann BCs (zero-gradient): use 'replicate' padding, which repeats edge values
            #u_padded = F.pad(u, pad=(1, 1, 1, 1), mode='replicate')
            # Apply convolution (i.e., apply Laplacian kernel to padded input)
            #lap = F.conv2d(u_padded, kernel)

            # No padding calculation (interior points only) 
            lap = F.conv2d(u, kernel, padding=0)

            return lap
      

    ## ----------------------------------------------------------------------- ##
    #  PDE loss calculation: Residuals 
    ## ----------------------------------------------------------------------- ##
    
    def residual_finite_difference(self, y_pred):
        """
        Finite-difference physics-informed loss for AP model: 
        (Including conversion of units and accountance for the resolution of the data)

        Computes the losses on a per sample basis and then either sums OR averages across the batch. 

        Parameters
        ----------
        y_pred : Output tensor of shape [B, 2, T, H, W]  
    
        Returns
        -------
        loss : 
            Physics loss for the PDE residuals computed via finite difference method.
        """

        B, C, T_pred, H, W = y_pred.shape
        assert C == 2, "Expected 2 channels (V, W)"
  
        V_pred = y_pred[:, 0] # [B, T, H, W]
        W_pred = y_pred[:, 1] # [B, T, H, W]
 
        print(f"V_pred shape: {V_pred.shape}")
        #print(f"V_pred: {V_pred[0, 0]}")
        #print(f"W_pred shape: {W_pred.shape}")
        
        # Convert delta_t to au
        delta_t_au = self.delta_t / self.t_scale
        print(f"delta_t_au value: {delta_t_au}")

        # Iterate over all available frames in the sequence 
        res_loss = 0.0
        res_loss_per_sample = torch.zeros(B, device=V_pred.device)  # per-sample residuals

        for i in range(B):  # loop over samples in the batch
            
            sample_res_loss = 0.0

            # Extract the first frame of the sample to use in the time difference calculations
            V_prev_i = V_pred[i, 0]  # [H, W]
            W_prev_i = W_pred[i, 0]

            # Convert the voltage units according to the AP cell model
            #V_prev_i = (V_prev_i - self.V_rest) / self.V_amp

            #print(f"V_pred value: {V_prev_i}")

            #print(f"V_prev_i shape: {V_prev_i.shape}")
            #print(f"W_prev_i shape: {W_prev_i.shape}")
            
            for t in range(T_pred - 1):
                #print(t+1)
                V_curr_i = V_pred[i, t+1 ]  # [H, W]
                W_curr_i = W_pred[i, t+1 ]
                #print(f"V_curr_i shape: {V_curr_i.shape}")
                #print(f"W_curr_i shape: {W_curr_i.shape}")

                # Convert the voltage units according to the AP cell model
                print(f"V_curr_i: {V_curr_i}")
                #V_curr_i = (V_curr_i - self.V_rest) / self.V_amp
                #print(f"V_curr_i rescaled to AU: {V_curr_i}")

                # Time derivatives 
                #print(f"Calculating Time Derivatives with dt = {delta_t_au}")
                V_t_i = (V_curr_i - V_prev_i) / delta_t_au
                W_t_i = (W_curr_i - W_prev_i) / delta_t_au

                #print(f"V_curr_i shape: {V_curr_i.shape}")
                #print(f"W_curr_i shape: {W_curr_i.shape}")

                # Laplacian (interior points)
               
                V_curr_ = V_curr_i.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W] for laplacian
                print(f"V_curr shape: {V_curr_.shape}")
                lap_V = self.laplacian_2d(V_curr_).squeeze()  # [H-2, W-2]
                print(f"Laplacian Shape: {lap_V.shape}")

                # Crop variables to match Laplacian shape
                V_interior = V_curr_i[1:-1, 1:-1]
                W_interior = W_curr_i[1:-1, 1:-1]
                V_t_interior = V_t_i[1:-1, 1:-1]
                W_t_interior = W_t_i[1:-1, 1:-1]

                print(f"V_interior shape: {V_interior.shape}")
                print(f"W_interior shape: {W_interior.shape}")
                print(f"V_t_interior shape: {V_t_interior.shape}")
                print(f"W_t_interior shape: {W_t_interior.shape}")

                # AP model RHS
                rhs_V = self.D * lap_V - self.k * V_interior * (V_interior - self.a) * (V_interior - 1) - V_interior * W_interior
                rhs_W = (self.epsilon + self.mu1 * W_interior / (V_interior + self.mu2)) * (-W_interior - self.k * V_interior * (V_interior - self.b - 1))
                
                # AP model RHS (with BCs included)
                #rhs_V = self.D * lap_V - self.k * V_curr_i * ( V_curr_i - self.a) * ( V_curr_i - 1) -  V_curr_i * W_curr_i
                #rhs_W = (self.epsilon + self.mu1 * W_curr_i / ( V_curr_i + self.mu2)) * (-W_curr_i - self.k *  V_curr_i * ( V_curr_i - self.b - 1))
                
                
                # print(f"rhs_V shape: {rhs_V.shape}, rhs = {rhs_V}")
                # print(f"rhs_W shape: {rhs_W.shape}, rhs = {rhs_W}")
                
                # Residual losses (mean over interior points)
                loss_V = self.loss_fn(V_t_interior, rhs_V, reduction=self.reduction)
                loss_W = self.loss_fn(W_t_interior, rhs_W, reduction=self.reduction)

                # Residual loss (BCs included)
                #loss_V = self.loss_fn(V_t_i, rhs_V, reduction=self.reduction)
                #loss_W = self.loss_fn(W_t_i, rhs_W, reduction=self.reduction)

                #print(f"loss_V shape: {loss_V.shape}, loss = {loss_V}")
                #print(f"loss_W shape: {loss_W.shape}, loss = {loss_W}")

                sample_res_loss += self.v_loss_weighting * loss_V + self.w_loss_weighting * loss_W

                # Update previous frame
                V_prev_i, W_prev_i = V_curr_i, W_curr_i

            # Average over frames for this sample
            res_loss_per_sample[i] = sample_res_loss / (T_pred - 1)
            #print(f"Residal loss per sample shape: {res_loss_per_sample[i].shape}")
            #print(f"Residal loss per sample: {res_loss_per_sample[i]}")
        
        if self.reduction == 'mean':
            # Average over batch
            res_loss = res_loss_per_sample.mean()
            #print(f"Residal loss per batch (mean): {res_loss_per_sample.mean()}")
        elif self.reduction == 'sum':
            # Sum over batch
            res_loss = res_loss_per_sample.sum()
            #print(f"Residal loss per batch (sum): {res_loss_per_sample.sum()}")

        return res_loss


    def residual_finite_difference_NEW(self, y_pred):
            """
            New FDM for residual calculation: Uses inbuilt FDM calculator to compute the laplacian for V
            Uses central FDM for the dVDt dWdt calculations
            Averages all of the losses over the spatiotemporal space AND over the batch. 

            Note that the time difference between frames is very coarse so the results could end up being a bit clunky. 
            Also note that dx dy are computed using the grid resolution. 
            Parameters
            ----------
            y_pred : Output tensor of shape [B, 2, T, H, W]  
        
            Returns
            -------
            loss : 
                Physics loss for the PDE residuals computed via finite difference method.
            """

            B, C, T_pred, H, W = y_pred.shape
            assert C == 2, "Expected 2 channels (V, W)"
            # Extract the Voltage and Recovery channels 
            V_pred = y_pred[:, 0] # [B, ch = 0, T, H, W]
            W_pred = y_pred[:, 1] # [B, ch = 1, T, H, W]

            # Verify the shape of each field tensor 
            print(f"V_pred shape: {V_pred.shape}")
            print(f"W_pred shape: {W_pred.shape}")

            # Establish 2D finite differences for calculating the laplacian for V:
            # Compute grid spacings from the tensor resolution - all are representative of a 10cm x 10cm slab
            dx = self.Lx / (W -1)
            dy = self.Ly / (H -1)
            # Establish time scaling:
            dt = self.Lt / (T_pred -1)
            dt = dt / self.t_scale

            print(f"dx={dx:.4f}, dy={dy:.4f}, dt={dt:.4f}")

            fd2d = FiniteDiff(dim=2, h=(dx, dy), periodic_in_x=True, periodic_in_y=True)
            
            '''
            fd3d = FiniteDiff(dim=3, h=(dt, dx, dy), periodic_in_x=False, periodic_in_y=True, periodic_in_z=True)
            # Compute the time derivatives:
            dVdt = fd3d.dx(V_pred)
            dWdt = fd3d.dx(W_pred)

            # Compute The Laplacian for V:
            dVdxx = fd2d.dx(V_pred, order=2)
            dVdyy = fd2d.dy(V_pred, order=2)
            lap_V = dVdxx + dVdyy 
            
            '''
            
            # Initialize accumulators 
            res_loss = torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)
            # Loop over batch samples # 
            for b in range(B): 
                Vb = V_pred[b] # [T, H, W] 
                Wb = W_pred[b] # [T, H, W] 
                # Central finite diff in time -> shape [T-2, H, W] 
                dVdt = (Vb[2:] - Vb[:-2]) / (2 * dt) 
                dWdt = (Wb[2:] - Wb[:-2]) / (2 * dt) 
                # Spatial Laplacian for V (same centered time indices) 
                lap_V = torch.zeros_like(dVdt) 
                for t in range(1, T_pred - 1): 
                    lap_V[t - 1] = fd2d.laplacian(Vb[t]) 
                    
                # Core variables at matching time indices 
                Vc = Vb[1:-1] 
                Wc = Wb[1:-1]

                # AP model RHS
                rhs_V = self.D * lap_V - self.k * Vc * (Vc- self.a) * (Vc - 1) - Vc* Wc
                rhs_W = (self.epsilon + self.mu1 * Wc / (Vc + self.mu2)) * (-Wc- self.k * Vc* (Vc - self.b - 1))
                
                # Residual losses (mean over interior points)
                loss_V = self.loss_fn(dVdt, rhs_V, reduction=self.reduction)
                loss_W = self.loss_fn(dWdt, rhs_W, reduction=self.reduction)

                res_loss += self.v_loss_weighting * loss_V + self.w_loss_weighting * loss_W

            # Average over batch
            #res_loss = res_loss.mean()
            #print(f"Residal loss per batch (mean): {res_loss.mean()}")

            # Normalise the loss over spatio-temporal points AND the batch
            T_centered = T_pred - 2
            res_loss = res_loss / (B * T_centered * H * W)

            return res_loss


# STACKED version of the data doesn't account for the overlapping windows and so doesn't represent the trajectory correctly. 
# This function is DEPRECIATED for now, but can be revisted if data structuring was altered later. 
    def residual_finite_difference_vectorised(self, y_pred):
        """
        Finite-difference physics-informed loss for AP model: 
        (Including conversion of units and accountance for the resolution of the data)

        Computes the PDE residual losses on a full batch by concatenating the samples to from a full trajectory. 

        Parameters
        ----------
        y_pred : Output tensor of shape [B, 2, T, H, W]  
    
        Returns
        -------
        loss : 
            Physics loss for the PDE residuals computed via finite difference method.

        Note: Due to the way that the data is constructed, samples represent a 'clip' of five time frames. 
        The data is trained on a consecutive input 'clips' and asked to predict the immediate next 'clip' from each of them. 
        Theorectically, output clips should also be consecutive clips and so can be concatenated to form a trajectory.
        This full trajectory of a batch can then be used to calculate the PDE loss. 

        Options: Combine the 5 frames of a single sample to from a trajectory
                Combine all samples in a batch to from a trajectory of batch size * 5 timesteps for a longer simulation. 
        """

        B, C, T, H, W = y_pred.shape
        assert C == 2, "Expected 2 channels (V, W)"
  
        V_pred = y_pred[:, 0] # [B, T, H, W]
        W_pred = y_pred[:, 1] # [B, T, H, W]
        
        # Convert the units of the voltage channel according to the AP cell model - only apply to spatial dimension:
        offset = torch.full((1, 1, H, W), 80.0, device=V_pred.device, dtype=V_pred.dtype)
        scale  = torch.full((1, 1, H, W), 100.0, device=V_pred.device, dtype=V_pred.dtype)
        V_pred = (V_pred + offset) / scale  # [B, T, H, W]

        # Convert delta_t to put time units into AU for the AP cell model and calculate the time derivatives
        delta_t_au = self.delta_t / 12.9
        V_t = (V_pred[:, 1:] - V_pred[:, :-1]) / delta_t_au  # [B, T-1, H, W]
        W_t = (W_pred[:, 1:] - W_pred[:, :-1]) / delta_t_au  # [B, T-1, H, W]

        # Combine batch and time for Laplacian calculations
        V_curr = V_pred[:, 1:]   # [B, T-1, H, W]
        V_curr_reshaped = V_curr.reshape(-1, 1, H, W)  # [(B*(T-1)), 1, H, W]
        lap_V = self.laplacian_2d(V_curr_reshaped)  # [(B*(T-1)), 1, H-2, W-2]
        lap_V = lap_V.view(B, T-1, H-2, W-2)

        # Restrict to the interior
        V_interior   = V_curr[:, :, 1:-1, 1:-1]    # [B, T-1, H-2, W-2]
        W_interior   = W_pred[:, 1:, 1:-1, 1:-1]   # [B, T-1, H-2, W-2]
        V_t_interior = V_t[:, :, 1:-1, 1:-1]
        W_t_interior = W_t[:, :, 1:-1, 1:-1]

        # Parallel calculations of the residuals
        rhs_V = (self.D * lap_V
                - self.k * V_interior * (V_interior - self.a) * (V_interior - 1)
                - V_interior * W_interior)

        rhs_W = ((self.epsilon + self.mu1 * W_interior / (V_interior + self.mu2)) *
                (-W_interior - self.k * V_interior * (V_interior - self.b - 1)))

        # Calculate the mean residual losses over the batch
        loss_V = self.loss_fn(V_t_interior, rhs_V, reduction=self.reduction)
        loss_W = self.loss_fn(W_t_interior, rhs_W, reduction=self.reduction)
        res_loss = self.v_loss_weighting * loss_V + self.w_loss_weighting * loss_W

        return res_loss
    
    def residual_query(self, y_pred, y_input, y_interior_query):
        """
        Query point based PDE residual loss for Aliev-Panfilov model.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted fields [B, 2, T, H, W]
        y_input : torch.Tensor
            Input sequence [B, 2, T_in, H, W]
        query_points : torch.Tensor
            Query points (normalized) for interpolation [B, N, 2] or [N, 2]
        t_indices : list or tensor
            Indices of timesteps to sample residual on (e.g. [0, 1, 2, 3, 4])

        Returns
        -------
        loss : torch.Tensor
            Residual loss averaged over query points and timesteps
        """

        B, C, T, H, W = y_pred.shape
        assert C == 2, "Expected [V, W] channels"


        # Compute grid spacings
        dx = self.Lx / (W - 1)
        dy = self.Ly / (H - 1)

        # Sample the query points from the grid:
        query_points_norm = y_interior_query.unsqueeze(2).detach().requires_grad_(True)  # [B, N, 1, 2]

        #Initialise the loss:
        res_loss = 0.0 

        for t in range(T): 
            # Establish the grid for selection on query points: 
            V_grid = y_pred[:, 0, t].unsqueeze(1)  # [B,1,H,W]
            W_grid = y_pred[:, 1, t].unsqueeze(1)
    
            # Sample at query points
            V_q = F.grid_sample(V_grid, query_points_norm, align_corners=True).squeeze(2).squeeze(1)  # [B,N]
            W_q = F.grid_sample(W_grid, query_points_norm, align_corners=True).squeeze(2).squeeze(1)  # [B,N]
        
            # second derivatives for Laplacian - using FDM and interpolation over the query points
            V_lap_grid = self.laplacian_2d(V_grid, dx=dx, dy=dy)
            lap_V = F.grid_sample(V_lap_grid, query_points_norm, align_corners=True).squeeze(2).squeeze(1)  # [B,N]
            
            '''
            # Use autograd to compute the spatial derivatives
            V_q_sum = V_q.sum()
            grads = torch.autograd.grad(V_q_sum, query_points_norm,
                                        create_graph=True, retain_graph=True)[0]  # [B,1,N,2]
            grads = grads.squeeze(2)  # [B,N,2]
            V_x = grads[..., 0] * (2.0 / (W-1) / self.dx)
            V_y = grads[..., 1] * (2.0 / (H-1) / self.dy)
            V_x_sum = V_x.sum()
            V_y_sum = V_y.sum()
            V_xx = torch.autograd.grad(V_x_sum, query_points_norm,
                                    create_graph=True, retain_graph=True)[0][..., 0].squeeze()
            V_yy = torch.autograd.grad(V_y_sum, query_points_norm,
                                    create_graph=True, retain_graph=True)[0][..., 1].squeeze()

            lap_V = V_xx + V_yy
            '''

            # --- Compute time derivative (using previous frame) ---
            if t == 0 and y_input is not None:
                V_prev = y_input[:, 0, -1]  # last observed frame
                W_prev = y_input[:, 1, -1]  # last observed frame
            else:
                V_prev = y_pred[:, 0, t-1]
                W_prev = y_pred[:, 1, t-1]

            V_prev_q = F.grid_sample(V_prev.unsqueeze(1), query_points_norm, align_corners=True).squeeze(2).squeeze(1)
            V_t = (V_q - V_prev_q) / (self.delta_t / 12.9)

            W_prev_q = F.grid_sample(W_prev.unsqueeze(1), query_points_norm, align_corners=True).squeeze(2).squeeze(1)
            W_t = (W_q - W_prev_q) / (self.delta_t / 12.9)

            # AP model RHS
            rhs_V = self.D * lap_V - self.k * V_q * (V_q - self.a) * (V_q - 1) - V_q * W_q
            rhs_W = (self.epsilon + self.mu1 * W_q / (V_q + self.mu2)) * (-W_q - self.k * V_q * (V_q - self.b - 1))

            # Residual losses 
            loss_V = self.loss_fn(V_t, rhs_V, reduction='mean')
            loss_W = self.loss_fn(W_t, rhs_W, reduction='mean')
            
            res_loss += self.v_loss_weighting * loss_V + self.w_loss_weighting * loss_W


        res_loss /= T
        return res_loss


    ## ----------------------------------------------------------------------- ##
    #  PDE loss calculation: Boundary Conditions
    ## ----------------------------------------------------------------------- ##
    def AP_BC_no_flux(self, y_pred):
        """
        Enforce zero Neumann BCs: dV/dn = 0 on all boundaries
       
        Parameters
        ----------
        y_pred : torch.Tensor
            [B, C, T, H, W] predicted output
        """
        V_pred = y_pred[:, 0]  # [B, T, H, W]
        B, T, H, W = V_pred.shape

        # Compute grid spacings
        dx = self.Lx / (W - 1)
        dy = self.Ly / (H - 1)

        bc_loss_total = 0.0        
        #NEW: Adjusting the BCs to second order (to match the laplacian)

        for t in range(T):
            V_curr = V_pred[:, t]  # [B, H, W]

            # --- Left/Right boundaries (∂V/∂x) ---
            left_grad  = (-3 * V_curr[:, :, 0] + 4 * V_curr[:, :, 1] - V_curr[:, :, 2]) / (2 * dx)
            right_grad = (3 * V_curr[:, :, -1] - 4 * V_curr[:, :, -2] + V_curr[:, :, -3]) / (2 * dx)

            # --- Top/Bottom boundaries (∂V/∂y) ---
            top_grad   = (-3 * V_curr[:, 0, :] + 4 * V_curr[:, 1, :] - V_curr[:, 2, :]) / (2 * dy)
            bottom_grad= (3 * V_curr[:, -1, :] - 4 * V_curr[:, -2, :] + V_curr[:, -3, :]) / (2 * dy)

            # Mean squared gradient penalties
            loss_left  = torch.mean(left_grad ** 2)
            loss_right = torch.mean(right_grad ** 2)
            loss_top   = torch.mean(top_grad ** 2)
            loss_bottom= torch.mean(bottom_grad ** 2)

            # Average BC loss for this frame
            bc_loss_total += (loss_left + loss_right + loss_top + loss_bottom) / 4.0
    
        # Average over all predicted frames
        bc_loss_total /= T
        return bc_loss_total
    
    def AP_BCs_query(self, y_pred, out_sub_level, y, boundary_points):
        """
        Using provided query points on the boundary, minimise the loss on the boundary points using the ground truth data
        
        Parameters
        ----------
        y_pred : torch.Tensor
            [B, C, T, H, W] predicted output
        boundary_points : dict
            {
            "left":   [B, n_left, 2]   (normalized coords)
            "right":  [B, n_right, 2]
            "top":    [B, n_top, 2]
            "bottom": [B, n_bottom, 2]
            }
        """
        self.counter = 0

        num_boundary = int(boundary_points.item() * out_sub_level)
        boundary_pred = y_pred.squeeze(0).squeeze(-1)[:num_boundary]
        y_bound = y.squeeze(0).squeeze(-1)[:num_boundary]
        
        assert boundary_pred.shape == y_bound.shape

        bc_loss = self.loss(boundary_pred, y_bound)

        return bc_loss
    ## ----------------------------------------------------------------------- ##
    #  Initial Condition Losses  
    ## ----------------------------------------------------------------------- ##
    def AP_IC_loss_input(self, y_pred, x):
        """
        Compute initial condition loss for the first predicted frame.
        
        y_pred: [B, 2, T_pred, H, W] 
        x: [B, 2, T_in, H, W] last observed frame used as IC

        V is scaled to the dimensionless units for the AP equation here. (makes it the same OOM as the W component)
        """

        # Use the last frame of input as IC
        V_true0 = x[:, 0, -1]
        #V_true0 = (y_input[:, 0, -1] + 80.0) / 100.0
        W_true0 = x[:, 1, -1]
        # Use the first frame of the prediction for comparison
        V_pred0 = y_pred[:, 0, 0]
        #V_pred0 = (y_pred[:, 0, 0] + 80.0) / 100.0
        W_pred0 = y_pred[:, 1, 0]

        loss_V = self.loss_fn(V_pred0, V_true0, reduction='mean')
        loss_W = self.loss_fn(W_pred0, W_true0, reduction='mean')

        ic_loss = self.v_loss_weighting * loss_V + self.w_loss_weighting * loss_W
        return ic_loss
    
    def AP_IC_loss_output(self, y_pred, y):
        """
        Compute initial condition loss for the first predicted frame.
        
        y_pred: [B, 2, T_pred, H, W] 
        y_input: [B, 2, T_in, H, W] 

        This function only compares the voltage channel. Scaling to normalise the values is optional...
        Compare the prediction to the GT data here.
        """

        # Use the first frame of GT output as IC
        V_true0 = y[:, 0, 0]
        # Use the first frame of the prediction for comparison
        V_pred0 = y_pred[:, 0, 0]

        loss_V = self.loss_fn(V_pred0, V_true0, reduction='mean')

        ic_loss = loss_V

        return ic_loss

    ## ----------------------------------------------------------------------- ##
    #  Combining loss functions
    ## ----------------------------------------------------------------------- ##

    def finite_difference_AP(self, y_pred, y, x):
        """
        Compute the weighted sum of the physics losses.

        Parameters
        ----------
        y_pred : torch.Tensor
            Model prediction of shape [B, 2, T, H, W]
        y : torch.Tensor
            Ground Truth data of shape [B, 2, T, H, W]
        x : torch.Tensor
            Ground Truth INPUT data of shape [B, 2, T, H, W]
        ic_weighting, bc_weighting, res_weighting : floats, optional
            Weighting between the physics losses. Default to equal weighting

        Returns
        -------
        ap_loss : torch.Tensor
           weighted sum of the losses for the physics model
        """

        ic_loss = self.AP_IC_loss_output(y_pred, y)
        bc_loss = self.AP_BC_no_flux(y_pred)
        res_loss = self.residual_finite_difference_NEW(y_pred)
        
        #print("---Physics Losses---")
        #print(f"Residuals (FDM): {res_loss}")


        # Weighted Combination
        #ap_loss = (self.ic_weighting * ic_loss) + (self.bc_weighting * bc_loss) + (self.res_weighting * res_loss)
        
        #Residuals Only
        ap_loss = res_loss

        #print(f"Overall Physics Loss: {ap_loss}")
        #print("-------------------")
        
        return ap_loss
    
    def query_point_AP(self, y_pred, y_input, y_interior_query, y_boundary_query):
        """
        Compute the weighted sum of the physics losses.

        Parameters
        ----------
        y_pred : torch.Tensor
            Model prediction of shape [B, 2, T, H, W]
        y_input : torch.Tensor
            Ground INPUT truth data of shape [B, 2, T, H, W]
        ic_weighting, bc_weighting, res_weighting : floats, optional
            Weighting between the physics losses. Default to equal weighting

        Returns
        -------
        ap_loss : torch.Tensor
           weighted sum of the losses for the physics model
        """

        ic_loss = self.AP_IC_loss(y_pred, y_input)
        bc_loss = self.AP_BCs(y_pred, y_boundary_query)
        res_loss = self.residual_query(y_pred, y_input, y_interior_query)
        
        print("---Physics Losses---")
        print(f"ICs: {ic_loss}")
        print(f"BCS: {bc_loss}")
        print(f"Residuals (query points): {res_loss}")
        print("-------------------")

        # Weighted combination
        ap_loss = self.ic_weighting * ic_loss + self.bc_weighting * bc_loss + self.res_weighting * res_loss

        return ap_loss

    ## ----------------------------------------------------------------------- ##
    # Running the loss model 
    ## ----------------------------------------------------------------------- ##

    # Load the tensors (y_pred, y, x) that contain the data from the V and W channels. 
    def __call__(self, y_pred, y, x, y_interior_query, y_boundary_query, **kwargs):
        if self.method == "query_point":
            return self.query_point_AP(y_pred, y, y_interior_query, y_boundary_query)
        elif self.method == "finite_difference":
            return self.finite_difference_AP(y_pred, y, x)
        else:
            raise NotImplementedError("Specified method is not implemented")

## NEW: Class for calculating the AP physics loss using spectral derivatives 

class APFFTLoss(object):
    # Defining the AP model parameters:
    def __init__(self, 
                 D=1.0, k=8.0, a=0.15, epsilon=0.002, mu1=0.2, mu2=0.3, b=0.15,
                 Lx = 10.0,
                 Ly = 10.0,
                 Lt = 50,
                 t_scale = 12.9,
                 v_loss_weighting=1.0, 
                 w_loss_weighting=1.0,
                 device = 'cuda'
                 ):
        super().__init__()
    
        ## Assign the parameters ##
        self.device = device

        #PDE parameters
        self.D = D
        self.k = k
        self.a = a
        self.epsilon = epsilon
        self.mu1 = mu1
        self.mu2 = mu2
        self.b = b
        self.t_scale = t_scale
       
        # Loss calculation parameters
        self.v_loss_weighting = v_loss_weighting
        self.w_loss_weighting = w_loss_weighting 
 
        # Coordinate calculation parameters 
        self.Lx = Lx
        self.Ly = Ly
        self.Lt = Lt

    def FFT_FDM_AP_res(self, u):

        B, C, T, H, W = u.shape
        assert C == 2, "Expected 2 channels (V, W)"
        # Extract the Voltage and Recovery channels 
        V_pred = u[:, 0] # [B, T, H, W]
        W_pred = u[:, 1] # [B, T, H, W]

        # Verify the shape of each field tensor 
        print(f"V_pred shape: {V_pred.shape}")
        print(f"W_pred shape: {W_pred.shape}")

        # Permute to have time in the z direction: [B, H, W, T]
        V_pred = V_pred.permute(0, 2, 3, 1)
        W_pred = W_pred.permute(0, 2, 3, 1)

        batchsize = V_pred.size(0)
        nx = V_pred.size(1)
        ny = V_pred.size(2)
        nt = V_pred.size(3)

        u = V_pred.reshape(batchsize, nx, ny, nt)
        v = W_pred.reshape(batchsize, nx, ny, nt)

        #spatio temporal scaling
        dx = self.Lx / (nx)
        dt = self.Lt / (nt-1)
        # Establish time scaling (AP unit conversion)
        dt = dt / self.t_scale

        u_h = torch.fft.fftn(u, dim=[1, 2])
        v_h = torch.fft.fftn(v, dim=[1, 2])

        # Calculate FFT derivatives 
        # Wavenumbers in y-direction
        k_max = nx//2
        N = nx
        k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=self.device),
                        torch.arange(start=-k_max, end=0, step=1, device=self.device)), 0).reshape(N, 1).repeat(1, N).reshape(1,N,N,1)
        k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=self.device),
                        torch.arange(start=-k_max, end=0, step=1, device=self.device)), 0).reshape(1, N).repeat(N, 1).reshape(1,N,N,1)
        ux_h = 2j *np.pi*k_x*u_h
        uxx_h = 2j *np.pi*k_x*ux_h
        uy_h = 2j *np.pi*k_y*u_h
        uyy_h = 2j *np.pi*k_y*uy_h

        uxx = torch.fft.irfftn(uxx_h[:, :, :k_max+1], dim=[1, 2])
        uyy = torch.fft.irfftn(uyy_h[:, :, :k_max+1], dim=[1, 2])
        ut = (u[..., 2:] - u[..., :-2]) / (2 * dt)
    
        vt = (v[..., 2:] - v[..., :-2]) / (2 * dt)

        #Plug in the residuals to AP model 
        Du = ut - self.D*(uxx + uyy) + self.k*u*(u-self.a)*(u -1) - u*v[..., 1:-1]
        Dv = vt - (self.epsilon + self.mu1 * v / (v + self.mu2)) * (-v - self.k * u * (u - self.b - 1))
        Dv = Dv[..., 1:-1]
    
        return Du, Dv
   
    def FFT_res(self, u):
        B, C, T, H, W = u.shape
        assert C == 2, "Expected 2 channels (V, W)"
        V_pred, W_pred = u[:, 0], u[:, 1]  # [B, T, H, W]

        V_pred = V_pred.permute(0, 2, 3, 1)  # [B, H, W, T]
        W_pred = W_pred.permute(0, 2, 3, 1)

        nx, ny, nt = V_pred.shape[1], V_pred.shape[2], V_pred.shape[3]
        dx = self.Lx / nx
        dt = (self.Lt / (nt - 1)) / self.t_scale

        # FFT wavenumbers
        kx = torch.fft.fftfreq(nx, d=dx).to(self.device).reshape(nx, 1).repeat(1, ny)
        ky = torch.fft.fftfreq(ny, d=dx).to(self.device).reshape(1, ny).repeat(nx, 1)

        # FFT of fields
        u_h = torch.fft.fftn(V_pred, dim=[1, 2])
        v_h = torch.fft.fftn(W_pred, dim=[1, 2])

        # Derivatives in spectral domain
        j = 1j
        uxx_h = -(2 * np.pi) ** 2 * (kx ** 2)[None, :, :, None] * u_h
        uyy_h = -(2 * np.pi) ** 2 * (ky ** 2)[None, :, :, None] * u_h
        vxx_h = -(2 * np.pi) ** 2 * (kx ** 2)[None, :, :, None] * v_h
        vyy_h = -(2 * np.pi) ** 2 * (ky ** 2)[None, :, :, None] * v_h

        # Back to real space
        uxx = torch.fft.ifftn(uxx_h, dim=[1, 2]).real
        uyy = torch.fft.ifftn(uyy_h, dim=[1, 2]).real
        vxx = torch.fft.ifftn(vxx_h, dim=[1, 2]).real
        vyy = torch.fft.ifftn(vyy_h, dim=[1, 2]).real

        # Time derivatives (central difference)
        ut = (V_pred[..., 2:] - V_pred[..., :-2]) / (2 * dt)
        vt = (W_pred[..., 2:] - W_pred[..., :-2]) / (2 * dt)


        # Mid-time slices
        u_mid, v_mid = V_pred[..., 1:-1], W_pred[..., 1:-1]


        # Clamp to avoid explosion
        u_mid = torch.clamp(u_mid, -10.0, 10.0)
        v_mid = torch.clamp(v_mid, -10.0, 10.0)

        # Safe denominator for v/(v+mu2)
        eps = 1e-8
        den = v_mid + self.mu2 + eps
        safe_frac = self.mu1 * v_mid / den

        # Residuals
        Du = ut - self.D * (uxx + uyy)[..., 1:-1] + self.k * u_mid * (u_mid - self.a) * (u_mid - 1) - u_mid * v_mid
        Dv = vt - (self.epsilon + safe_frac) * (-v_mid - self.k * u_mid * (u_mid - self.b - 1))

        # Debug prints (can comment out after first check)
        print("Du range:", Du.min().item(), Du.max().item())
        print("Dv range:", Dv.min().item(), Dv.max().item())
        print("u_mid range:", u_mid.min().item(), u_mid.max().item())
        print("v_mid range:", v_mid.min().item(), v_mid.max().item())
        print("dt:", dt)

        return Du, Dv
   
    ## ----------------------------------------------------------------------- ##
    # Running the PDE residual loss model 
    ## ----------------------------------------------------------------------- ##

    # Load the tensors (y_pred, y, x) that contain the data from the V and W channels. 
    def __call__(self, *, y_pred: torch.Tensor, **kwargs):
        """
        Compute the physics-based AP residual loss.
        Only y_pred is required; other args are ignored for compatibility.
        """
        if y_pred is None:
            raise ValueError("y_pred must be provided to compute APFFTLoss.")

        if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
            raise RuntimeError("y_pred contains NaN or Inf before residual calculation")


        Du, Dv = self.FFT_res(y_pred)
        loss_V = torch.mean(Du ** 2)
        loss_W = torch.mean(Dv ** 2)

        res_loss = self.v_loss_weighting * loss_V + self.w_loss_weighting * loss_W

        #print('--Physics Loss (FFT)--')
        #print(f"FFT residual loss: {res_loss.item():.6f}")

        return res_loss




        