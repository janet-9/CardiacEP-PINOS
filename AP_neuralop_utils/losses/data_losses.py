"""
data_losses.py contains code to compute standard data objective 
functions for training Neural Operators. 

By default, losses expect arguments y_pred (model predictions) and y (ground y.)
"""

import math
from typing import List
import torch
from .differentiation import FiniteDiff
import torch.nn.functional as F


#LP loss function from the neuralops library - used as the primary data loss
class LpLoss(object):
    """
    LpLoss provides the L-p norm between two 
    discretized d-dimensional functions. Note that 
    LpLoss always averages over the spatial dimensions.

    .. note :: 
        In function space, the Lp norm is an integral over the
        entire domain. To ensure the norm converges to the integral,
        we scale the matrix norm by quadrature weights along each spatial dimension.

        If no quadrature is passed at a call to LpLoss, we assume a regular 
        discretization and take ``1 / measure`` as the quadrature weights. 

    Parameters
    ----------
    d : int, optional
        dimension of data on which to compute, by default 1
    p : int, optional
        order of L-norm, by default 2
        L-p norm: [\\sum_{i=0}^n (x_i - y_i)**p] ** (1/p) 
    reduction : str, optional
        whether to reduce across the batch and channel dimensions
        by summing ('sum') or averaging ('mean')
        .. warning:: 

            ``LpLoss`` always reduces over the spatial dimensions according to ``self.measure``.
            `reduction` only applies to the batch and channel dimensions.
    
    eps : float, optional
        small number added to the denominator for numerical stability when using the relative loss

    v_loss_weighting, w_loss_weighting: weightings to apply to each channel when calculting the overall loss. 
     ..warning::
        Channel 0 represents the voltage data and Channel 1 represents the recovery current data (a latent variable)
    ```
    """

    def __init__(self, d=1, p=2, measure=1., reduction='sum', eps=1e-8, v_loss_weighting=1.0, 
                 w_loss_weighting=1.0,):
        super().__init__()

        self.d = d
        self.p = p
        self.eps = eps
        # channel weightings
        self.v_loss_weighting = v_loss_weighting
        self.w_loss_weighting = w_loss_weighting 
        
        allowed_reductions = ["sum", "mean"]
        assert reduction in allowed_reductions,\
        f"error: expected `reduction` to be one of {allowed_reductions}, got {reduction}"
        self.reduction = reduction

        if isinstance(measure, float):
            self.measure = [measure]*self.d
        else:
            self.measure = measure
    
    @property
    def name(self):
        return f"L{self.p}_{self.d}Dloss"
    
    def reduce_all(self, x):
        """
        reduce x across the batch according to `self.reduction`

        Params
        ------
        x: torch.Tensor
            inputs
        """
        if self.reduction == 'sum':
            x = torch.sum(x)
        else:
            x = torch.mean(x)
        
        return x

    def rel(self, x, y):
        """
        rel: relative LpLoss
        computes ||x-y||/(||y|| + eps)

        Parameters
        ----------
        x : torch.Tensor
            inputs
        y : torch.Tensor
            targets
        """
        diff = torch.norm(torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d), \
                          p=self.p, dim=-1, keepdim=False)
        ynorm = torch.norm(torch.flatten(y, start_dim=-self.d), p=self.p, dim=-1, keepdim=False)
        diff = diff/(ynorm + self.eps)

        diff = self.reduce_all(diff).squeeze()
        return diff

    def __call__(self, y_pred, y, **kwargs):
        return self.rel(y_pred, y)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     

#MSE based loss for boundary locations
class BoundaryLoss(object):
    """
    Boundary Loss calculates the difference between the ground truth and predicted tensors, only on the boundary locations. 
    It assumes a uniform 2D grid, with the boundary locations being the edges of this grid. 
    This loss is also designed to work with 2 channel data structure (V, W - for the AP equation) and calculates the boundary losses
    per channel and adds then in a weighted sum. 
    The reduction refers to a reduction across the batch dimension. 

    Parameters
    ----------
    reduction : str, optional
        whether to reduce across the batch dimension
        by summing ('sum') or averaging ('mean')

    v_loss_weighting, w_loss_weighting: weightings to apply to each channel when calculting the overall loss. 
     ..warning::
        Channel 0 represents the voltage data and Channel 1 represents the recovery current data (a latent variable)
    ```
    """

    def __init__(self, reduction='mean', v_loss_weighting=1.0, w_loss_weighting=1.0, loss_fn=F.mse_loss):
        super().__init__()

        # channel weightings
        self.v_loss_weighting = v_loss_weighting
        self.w_loss_weighting = w_loss_weighting 
        
        #reduction along time and batch dimensions
        allowed_reductions = ["sum", "mean"]
        assert reduction in allowed_reductions,\
        f"error: expected `reduction` to be one of {allowed_reductions}, got {reduction}"
        self.reduction = reduction

        # Definition loss function
        self.loss_fn = loss_fn
    
    @property
    def name(self):
        return f"Boundary_loss"
    
    def BCs(self, y_pred, y):
        """
        Enforce boundary condition agreement with the ground truth model
       
        Parameters
        ----------
        y_pred : torch.Tensor
            [B, C, T, H, W] predicted output

        y : torch.Tensor
        [B, C, T, H, W] ground truth 
        """
        V_pred = y_pred[:, 0]  # [B, T, H, W]
        B, T, H, W = V_pred.shape
        #print(f"V_pred shape: {V_pred.shape}")
        W_pred = y_pred[:, 1]  # [B, T, H, W]
        B, T, H, W = W_pred.shape
        #print(f"V_pred shape: {W_pred.shape}")

        V_gt = y[:, 0]  # [B, T, H, W]
        B, T, H, W = V_gt.shape
        #print(f"V_gt shape: {V_gt.shape}")
        W_gt = y[:, 1]  # [B, T, H, W]
        B, T, H, W = W_gt.shape
        #print(f"W_gt shape: {W_gt.shape}")

        bc_loss = 0.0   

        # Extract domain boundaries (outermost rows and columns)
        def get_boundaries_grid(x):
            top = x[:, :, 0, :]          # includes top-left/right
            bottom = x[:, :, -1, :]
            left = x[:, :, 1:-1, 0]      # skip corners
            right = x[:, :, 1:-1, -1]
            # Flatten each boundary and concatenate along the last dimension
            return torch.cat([
                top.reshape(x.shape[0], x.shape[1], -1),
                bottom.reshape(x.shape[0], x.shape[1], -1),
                left.reshape(x.shape[0], x.shape[1], -1),
                right.reshape(x.shape[0], x.shape[1], -1)
            ], dim=-1)

        V_pred_bound = get_boundaries_grid(V_pred)
        W_pred_bound = get_boundaries_grid(W_pred)
        V_gt_bound   = get_boundaries_grid(V_gt)
        W_gt_bound   = get_boundaries_grid(W_gt)

        #print(f"V_pred boundary: {V_pred_bound}")
        #print(f"W_pred boundary: {W_pred_bound}")
        #print(f"V_gt boundary: {V_gt_bound}")
        #print(f"W_gt boundary: {W_gt_bound}")

        # Minimise the difference between the GT and prediction on the boundary:
        loss_V = self.loss_fn(V_pred_bound, V_gt_bound, reduction= self.reduction)
        loss_W = self.loss_fn(W_pred_bound, W_gt_bound, reduction= self.reduction)

        #print(f"loss_V boundary shape: {loss_V.shape}, loss_V: {loss_V}")
        #print(f"loss_W boundary shape: {loss_W.shape}, loss_W: {loss_W}")

        bc_loss = self.v_loss_weighting * loss_V + self.w_loss_weighting * loss_W
        return bc_loss
     
    def __call__(self, y_pred, y, **kwargs):
        return self.BCs(y_pred, y)  
    
#NEW: RMSE loss incuded
class RMSELoss(object):
    """
    RMSELoss computes absolute root mean-squared L2 error between two tensors.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, *, y_pred: torch.Tensor, y: torch.Tensor, dim: List[int]=None, **kwargs):
        """RMSE loss call 

        Parameters
        ----------
        y_pred : torch.Tensor
            tensor of predictions
        y : torch.Tensor
            ground truth, must be same shape as y_pred
        dim : List[int], optional
            dimensions across which to compute RMSE, by default None
        """
        assert y_pred.shape == y.shape, (y.shape, y_pred.shape)
        if dim is None:
            dim = list(range(1, y_pred.ndim)) # no reduction across batch dim
        #return torch.sqrt(torch.mean((y_pred - y) ** 2, dim=dim)).sum() # sum of RMSEs for each element
        return torch.sqrt(torch.mean((y_pred - y) ** 2)) # Computes global RMSE for all samples 


#Initial condition loss for input-output pair training method:
class ICLoss(object):
    """
    Initial Conditions Loss calculates the difference between the ground truth and predicted tensors, at teh start of the predicted trajectory 
    It assumes a uniform 2D grid, with the boundary locations being the edges of this grid. 
    This loss is also designed to work with 2 channel data structure (V, W - for the AP equation) and calculates the boundary losses
    per channel and adds then in a weighted sum. 
    The reduction refers to a reduction across the batch dimension. 

    Parameters
    ----------
    reduction : str, optional
        whether to reduce across the batch dimension
        by summing ('sum') or averaging ('mean')

    v_loss_weighting, w_loss_weighting: weightings to apply to each channel when calculting the overall loss. 
     ..warning::
        Channel 0 represents the voltage data and Channel 1 represents the recovery current data (a latent variable)
    ```
    """

    def __init__(self, reduction='mean', v_loss_weighting=1.0, w_loss_weighting=1.0, loss_fn=F.mse_loss):
        super().__init__()

        # channel weightings
        self.v_loss_weighting = v_loss_weighting
        self.w_loss_weighting = w_loss_weighting 
        
        #reduction along time and batch dimensions
        allowed_reductions = ["sum", "mean"]
        assert reduction in allowed_reductions,\
        f"error: expected `reduction` to be one of {allowed_reductions}, got {reduction}"
        self.reduction = reduction

        # Definition loss function
        self.loss_fn = loss_fn
    
    @property
    def name(self):
        return f"IC_loss"
    
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
    
    def __call__(self, *, y_pred: torch.Tensor, y: torch.Tensor, dim: List[int]=None, **kwargs):
        """IC loss call 

        Parameters
        ----------
        y_pred : torch.Tensor
            tensor of predictions
        y : torch.Tensor
            ground truth, must be same shape as y_pred
        dim : List[int], optional
            dimensions across which to compute RMSE, by default None
        """
        assert y_pred.shape == y.shape, (y.shape, y_pred.shape)
        ic_loss = self.AP_IC_loss_output(y_pred, y)
        return ic_loss


#No flux boundary condition loss for input-output pair training method:
class BCNeumann(object):
    """
    Enforces No Flux Boundary conditions on the predicted tensor
    It assumes a uniform 2D grid, with the boundary locations being the edges of this grid. 
    This loss is also designed to work with 2 channel data structure (V, W - for the AP equation) and calculates the boundary losses
    per channel and adds then in a weighted sum. 
    The reduction refers to a reduction across the batch dimension. 

    Parameters
    ----------
    reduction : str, optional
        whether to reduce across the batch dimension
        by summing ('sum') or averaging ('mean')

    v_loss_weighting, w_loss_weighting: weightings to apply to each channel when calculting the overall loss. 
     ..warning::
        Channel 0 represents the voltage data and Channel 1 represents the recovery current data (a latent variable)
    ```
    """

    def __init__(self, reduction='mean', v_loss_weighting=1.0, w_loss_weighting=1.0, loss_fn=F.mse_loss, Lx = 10, Ly = 10):
        super().__init__()

        # channel weightings
        self.v_loss_weighting = v_loss_weighting
        self.w_loss_weighting = w_loss_weighting 

        #Grid dimensions 
        self.Lx = Lx
        self.Ly = Ly
        
        #reduction along time and batch dimensions
        allowed_reductions = ["sum", "mean"]
        assert reduction in allowed_reductions,\
        f"error: expected `reduction` to be one of {allowed_reductions}, got {reduction}"
        self.reduction = reduction

        # Definition loss function
        self.loss_fn = loss_fn
    
    @property
    def name(self):
        return f"BC_noflux_loss"
    
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
    
    
    def __call__(self, *, y_pred: torch.Tensor, y: torch.Tensor, dim: List[int]=None, **kwargs):
        """No Flux BC loss call 

        Parameters
        ----------
        y_pred : torch.Tensor
            tensor of predictions
        y : torch.Tensor
            ground truth, must be same shape as y_pred
        dim : List[int], optional
            dimensions across which to compute RMSE, by default None
        """
        assert y_pred.shape == y.shape, (y.shape, y_pred.shape)
        bc_loss = self.AP_BC_no_flux(y_pred)
        return bc_loss