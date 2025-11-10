import torch
import logging
from torch import nn
from typing import Dict, List, Optional, Callable
import inspect
logger = logging.getLogger(__name__)

'''
#NEW: Weighted Loss to handle the AP loss call: 
class WeightedSumLoss(object):
    """
    Computes an average or weighted sum of given losses.
    """

    def __init__(self, losses, weights=None):
        super().__init__()
        if weights is None:
            weights = [1.0 / len(losses)] * len(losses)
        if not len(weights) == len(losses):
            raise ValueError("Each loss must have a weight.")
        self.losses = list(zip(losses, weights))


    def __call__(self, *args, **kwargs):
            weighted_loss = 0.0
            for loss, weight in self.losses:
                sig = inspect.signature(loss.__call__)
                valid_keys = sig.parameters.keys()
                filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
                weighted_loss += weight * loss(*args, **filtered_kwargs)
            return weighted_loss

    def __str__(self):
        description = "Combined loss: "
        for loss, weight in self.losses:
            description += f"{loss} (weight: {weight}) "
        return description
'''

#NEW: Weighted Loss to handle the AP loss call: 
class WeightedSumLoss(object):
    """
    Computes a weighted sum of given losses with automatic signature handling.
    Works even if some losses only take y_pred, and others take (y_pred, y).
    """

    def __init__(self, losses, weights=None):
        super().__init__()
        if weights is None:
            weights = [1.0 / len(losses)] * len(losses)
        if len(weights) != len(losses):
            raise ValueError("Each loss must have a corresponding weight.")
        self.losses = list(zip(losses, weights))

    def __call__(self, y_pred, y=None, **kwargs):
        weighted_loss = 0.0
        for loss, weight in self.losses:
            sig = inspect.signature(loss.__call__)
            params = list(sig.parameters.keys())

            # Case 1: loss expects both y_pred and y
            if "y_pred" in params and "y" in params:
                loss_val = loss(y_pred=y_pred, y=y, **kwargs)

            # Case 2: loss only expects y_pred
            elif "y_pred" in params:
                loss_val = loss(y_pred=y_pred, **kwargs)

            # Case 3: loss expects positional args
            else:
                try:
                    loss_val = loss(y_pred, y, **kwargs)
                except TypeError:
                    loss_val = loss(y_pred, **kwargs)

            weighted_loss += weight * loss_val

        return weighted_loss

    def __str__(self):
        return " + ".join([f"{w}*{l}" for l, w in self.losses])


#NEW: Adding in a backbone loss for a fine-tuning phase
class OperatorBackboneLoss(object):
    def __init__(self, model, reduction="mean", relative=True, eps=1e-8):
        super().__init__()
        self.model = model
        self.backbone_params = {
            k: v.clone().detach()
            for k, v in model.state_dict().items()
            if isinstance(v, torch.Tensor)
        }
        self.reduction = reduction
        self.relative = relative
        self.eps = eps

    def __call__(self, **kwargs):
        losses = []
        for name, param in self.model.named_parameters():
            if name in self.backbone_params:
                ref = self.backbone_params[name].to(param.device)
                diff = param - ref
                if self.relative:
                    denom = ref.norm() + self.eps
                    losses.append((diff.norm() / denom) ** 2)
                else:
                    losses.append(torch.sum(diff.abs() ** 2))

        if not losses:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)

        loss = torch.stack(losses).mean() if self.reduction == "mean" else torch.stack(losses).sum()
        return loss

# NEW: Adding in an adaptive training loss :
class AdaptiveTrainingLoss(object):
    def __init__(self, aggregator, data_loss, phys_loss, bc_loss, ic_loss):
        super().__init__()
        self.aggregator = aggregator
        self.data_loss = data_loss
        self.phys_loss = phys_loss
        self.bc_loss = bc_loss
        self.ic_loss = ic_loss
        self.step = 0  # global step counter

    def __call__(self, y_pred, y=None, x=None, **kwargs):
        # --- Compute each sub-loss ---
        losses = {
            "data": self.data_loss(y_pred, y),      
            "phys": self.phys_loss(y_pred, x),      # PDE residual
            "ic":   self.ic_loss(y_pred, x),        # initial condition
            "bc":   self.bc_loss(y_pred, x)         # boundary condition
        }

        # --- Combine them adaptively ---
        total_loss, lambdas = self.aggregator(losses, step=self.step)
        self.step += 1

        # Optional monitoring
        if self.step % 100 == 0:
            print(f"[Adapt] Step {self.step}: λ = {lambdas.detach().cpu().numpy()}")

        return total_loss