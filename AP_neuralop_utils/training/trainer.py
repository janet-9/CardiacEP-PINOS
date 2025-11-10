from timeit import default_timer
from pathlib import Path
from typing import Union
import sys
import warnings
import inspect

import torch
from torch.cuda import amp
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# Only import wandb and use if installed
wandb_available = False
try:
    import wandb
    wandb_available = True
except ModuleNotFoundError:
    wandb_available = False
import json 
import os
import neuralop.mpu.comm as comm
from neuralop.losses import LpLoss, H1Loss, HdivLoss, MSELoss
from ..losses import APLoss, OperatorBackboneLoss, WeightedSumLoss
from .training_state import load_training_state, save_training_state
import math 

class Trainer:
    """
    A general Trainer class to train neural-operators on given datasets. 

    .. note ::
        Our Trainer expects datasets to provide batches as key-value dictionaries, ex.: 
        ``{'x': x, 'y': y}``, that are keyed to the arguments expected by models and losses. 
        For specifics and an example, check ``neuralop.data.datasets.DarcyDataset``. 

    Parameters
    ----------
    model : nn.Module
    n_epochs : int
    wandb_log : bool, default is False
        whether to log results to wandb
    device : torch.device, or str 'cpu' or 'cuda'
    mixed_precision : bool, default is False
        whether to use torch.autocast to compute mixed precision
    data_processor : DataProcessor class to transform data, default is None
        if not None, data from the loaders is transform first with data_processor.preprocess,
        then after getting an output from the model, that is transformed with data_processor.postprocess.
    eval_interval : int, default is 1
        how frequently to evaluate model and log training stats
    log_output : bool, default is False
        if True, and if wandb_log is also True, log output images to wandb
    use_distributed : bool, default is False
        whether to use DDP
    verbose : bool, default is False

    ==== AP PINOs adaptions ===
    We added json logging of the relevant information for later analysis. 
    """
    def __init__(
        self,
        *,
        model: nn.Module,
        n_epochs: int,
        wandb_log: bool=True,
        device: str='cpu',
        mixed_precision: bool=False,
        data_processor: nn.Module=None,
        eval_interval: int=1,
        log_output: bool=False,
        use_distributed: bool=False,
        verbose: bool=False,

    ):
        """
        """

        self.model = model
        self.n_epochs = n_epochs
        # only log to wandb if a run is active
        self.wandb_log = True
        if wandb_available:
            self.wandb_log = (wandb_log and wandb.run is not None)
        self.eval_interval = eval_interval
        self.log_output = log_output
        self.verbose = verbose
        self.use_distributed = use_distributed
        self.device = device
        # handle autocast device
        if isinstance(self.device, torch.device):
            self.autocast_device_type = self.device.type
        else:
            if "cuda" in self.device:
                self.autocast_device_type = "cuda"
            else:
                self.autocast_device_type = "cpu"
        self.mixed_precision = mixed_precision
        self.data_processor = data_processor
    
        # Track starting epoch for checkpointing/resuming
        self.start_epoch = 0

    def train(
        self,
        train_loader,
        test_loaders,
        optimizer,
        scheduler,
        regularizer=None,
        training_loss=None,
        eval_losses=None,
        save_every: int=None,
        save_best: int=None,
        save_dir: Union[str, Path]="./ckpt",
        resume_from_dir: Union[str, Path]=None,
        json_log_path: Union[str, Path]=None
    ):
        """Trains the given model on the given dataset.

        If a device is provided, the model and data processor are loaded to device here. 

        Parameters
        -----------
        train_loader: torch.utils.data.DataLoader
            training dataloader
        test_loaders: dict[torch.utils.data.DataLoader]
            testing dataloaders
        optimizer: torch.optim.Optimizer
            optimizer to use during training
        scheduler: torch.optim.lr_scheduler
            learning rate scheduler to use during training
        training_loss: training.losses function
            cost function to minimize
        eval_losses: dict[Loss]
            dict of losses to use in self.eval()
        save_every: int, optional, default is None
            if provided, interval at which to save checkpoints
        save_best: str, optional, default is None
            if provided, key of metric f"{loader_name}_{loss_name}"
            to monitor and save model with best eval result
            Overrides save_every and saves on eval_interval
        save_dir: str | Path, default "./ckpt"
            directory at which to save training states if
            save_every and/or save_best is provided
        resume_from_dir: str | Path, default None
            if provided, resumes training state (model, 
            optimizer, regularizer, scheduler) from state saved in
            `resume_from_dir`
        json_log_path = path to save the epoch metrics as a json file. Default is none. 
        
        Returns
        -------
        all_metrics: dict
            dictionary keyed f"{loader_name}_{loss_name}"
            of metric results for last validation epoch across
            all test_loaders
            
        """
        self.json_log_path = json_log_path


        self.optimizer = optimizer
        self.scheduler = scheduler
        if regularizer:
            self.regularizer = regularizer
        else:
            self.regularizer = None

        if training_loss is None:
            training_loss = LpLoss(d=2)
        
        # Warn the user if training loss is reducing across the batch
        if hasattr(training_loss, 'reduction'):
            if training_loss.reduction == "mean":
                warnings.warn(f"{training_loss.reduction=}. This means that the loss is "
                              "initialized to average across the batch dim. The Trainer "
                              "expects losses to sum across the batch dim.")

        if eval_losses is None:  # By default just evaluate on the training loss
            eval_losses = dict(l2=training_loss)
        
        # accumulated wandb metrics
        self.wandb_epoch_metrics = None
        #initialise the epoch metrics before training:
        epoch_metrics = None  

        # attributes for checkpointing
        self.save_every = save_every
        self.save_best = save_best
        if resume_from_dir is not None:
            self.resume_state_from_dir(resume_from_dir)

        # Load model and data_processor to device
        self.model = self.model.to(self.device)

        if self.use_distributed and dist.is_initialized():
            device_id = dist.get_rank()
            self.model = DDP(self.model, device_ids=[device_id], output_device=device_id)

        if self.data_processor is not None:
            self.data_processor = self.data_processor.to(self.device)
        
        # ensure save_best is a metric we collect
        if self.save_best is not None:
            metrics = []
            for name1, name2 in test_loaders.keys():
                for metric in eval_losses.keys():
                    metrics.append(f"({name1}, {name2})_{metric}")
                    #print(f"Metric for best performing model: {metrics}")
            assert self.save_best in metrics,\
                f"Error: expected a metric of the form <loader_name>_<metric>, got {save_best}"
            best_metric_value = float('inf')
            # either monitor metric or save on interval, exclusive for simplicity
            self.save_every = None

        if self.verbose:
            print(f'Training on {len(train_loader.dataset)} samples')
            print(f'Testing on {[len(loader.dataset) for loader in test_loaders.values()]} samples'
                  f'         on resolutions {[name for name in test_loaders]}.')
            sys.stdout.flush()
        
        for epoch in range(self.start_epoch, self.n_epochs):
            train_err, avg_loss, avg_lasso_loss, epoch_train_time =\
                  self.train_one_epoch(epoch, train_loader, training_loss)
            epoch_metrics = dict(
                train_err=train_err,
                avg_loss=avg_loss,
                avg_lasso_loss=avg_lasso_loss,
                epoch_train_time=epoch_train_time,
            )
            
            if epoch % self.eval_interval == 0:
                # evaluate and gather metrics across each loader in test_loaders
                eval_metrics = self.evaluate_all(epoch=epoch,
                                                eval_losses=eval_losses,
                                                test_loaders=test_loaders)

                epoch_metrics.update(**eval_metrics)
                # save checkpoint if conditions are met
                if save_best is not None:
                    if eval_metrics[save_best] < best_metric_value:
                        best_metric_value = eval_metrics[save_best]
                        self.checkpoint(save_dir)

            # save checkpoint if save_every and save_best is not set
            if self.save_every is not None:
                if epoch % self.save_every == 0:
                    self.checkpoint(save_dir)

            #save the training results in a json file - after transforming the tensor data:
            # Add epoch number to metrics
            epoch_metrics['epoch'] = epoch + 1

            def tensor_to_python(obj):
                if isinstance(obj, torch.Tensor):
                    if obj.numel() == 1:  # single value tensor
                        return obj.item()
                    else:  # tensor with multiple elements
                        return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: tensor_to_python(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [tensor_to_python(i) for i in obj]
                else:
                    return obj
                
            epoch_metrics_clean = tensor_to_python(epoch_metrics)

            # Append to JSON log file
            if self.json_log_path is not None:
                os.makedirs(os.path.dirname(self.json_log_path), exist_ok=True)
                with open(self.json_log_path, "a") as f:
                    f.write(json.dumps(epoch_metrics_clean) + "\n")

        return epoch_metrics
    

    def train_one_epoch(self, epoch, train_loader, training_loss):
        """train_one_epoch trains self.model on train_loader
        for one epoch and returns training metrics

        Parameters
        ----------
        epoch : int
            epoch number
        train_loader : torch.utils.data.DataLoader
            data loader of train examples
        test_loaders : dict
            dict of test torch.utils.data.DataLoader objects

        Returns
        -------
        all_errors
            dict of all eval metrics for the last epoch
        """
        self.on_epoch_start(epoch)
        avg_loss = 0
        avg_lasso_loss = 0
        self.model.train()
        if self.data_processor:
            self.data_processor.train()
        t1 = default_timer()
        train_err = 0.0
        
        # track number of training examples in batch
        self.n_samples = 0

        for idx, sample in enumerate(train_loader):
            
            loss = self.train_one_batch(idx, sample, training_loss)
            loss.backward()
            self.optimizer.step()

            train_err += loss.item()
            with torch.no_grad():
                avg_loss += loss.item()
                if self.regularizer:
                    avg_lasso_loss += self.regularizer.loss

        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(train_err)
        else:
            self.scheduler.step()

        epoch_train_time = default_timer() - t1

        train_err /= len(train_loader) # This gives the mean error for the epoch in terms of batches (total error/no. batches)
        avg_loss /= self.n_samples # This gives the mean error for the epoch in termss of samples (total error/no. samples)
        
        if self.regularizer:
            avg_lasso_loss /= self.n_samples
        else:
            avg_lasso_loss = None
        
        lr = None
        for pg in self.optimizer.param_groups:
            lr = pg["lr"]
        if self.verbose and epoch % self.eval_interval == 0:
            self.log_training(
                epoch=epoch,
                time=epoch_train_time,
                avg_loss=avg_loss,
                train_err=train_err,
                avg_lasso_loss=avg_lasso_loss,
                lr=lr,
               
            )
        return train_err, avg_loss, avg_lasso_loss, epoch_train_time

    def evaluate_all(self, epoch, eval_losses, test_loaders):
        # evaluate and gather metrics across each loader in test_loaders
        all_metrics = {}
        for loader_name, loader in test_loaders.items():
            loader_metrics = self.evaluate(eval_losses, loader,
                                    log_prefix=loader_name)   
            all_metrics.update(**loader_metrics)
        if self.verbose:
            self.log_eval(epoch=epoch,
                      eval_metrics=all_metrics)
        return all_metrics
    
    def evaluate(self, loss_dict, data_loader, log_prefix="", epoch=None):
        """Evaluates the model on a dictionary of losses

        Parameters
        ----------
        loss_dict : dict of functions
          each function takes as input a tuple (prediction, ground_truth)
          and returns the corresponding loss
        data_loader : data_loader to evaluate on
        log_prefix : str, default is ''
            if not '', used as prefix in output dictionary
        epoch : int | None
            current epoch. Used when logging both train and eval
            default None
        Returns
        -------
        errors : dict
            dict[f'{log_prefix}_{loss_name}] = loss for loss in loss_dict
        """
        # Ensure model and data processor are loaded to the proper device
        self.model = self.model.to(self.device)
        if self.data_processor is not None and self.data_processor.device != self.device:
            self.data_processor = self.data_processor.to(self.device)
        
        self.model.eval()
        if self.data_processor:
            self.data_processor.eval()

        errors = {f"{log_prefix}_{loss_name}": 0 for loss_name in loss_dict.keys()}

        # Warn the user if any of the eval losses is reducing across the batch
        for _, eval_loss in loss_dict.items():
            if hasattr(eval_loss, 'reduction'):
                if eval_loss.reduction == "mean":
                    warnings.warn(f"{eval_loss.reduction=}. This means that the loss is "
                                "initialized to average across the batch dim. The Trainer "
                                "expects losses to sum across the batch dim.")

        self.n_samples = 0
        with torch.no_grad():
            for idx, sample in enumerate(data_loader):
                return_output = False
                if idx == len(data_loader) - 1:
                    return_output = True
                eval_step_losses, outs = self.eval_one_batch(sample, loss_dict, return_output=return_output)

                for loss_name, val_loss in eval_step_losses.items():
                    errors[f"{log_prefix}_{loss_name}"] += val_loss
            
        for key in errors.keys():
            errors[key] /= self.n_samples

        # on last batch, log model outputs
        if self.log_output:
            errors[f"{log_prefix}_outputs"] = wandb.Image(outs)
        
        return errors
    
    def on_epoch_start(self, epoch):
        """on_epoch_start runs at the beginning
        of each training epoch. This method is a stub
        that can be overwritten in more complex cases.

        Parameters
        ----------
        epoch : int
            index of epoch

        Returns
        -------
        None
        """
        self.epoch = epoch
        return None

    def train_one_batch(self, idx, sample, training_loss):
        """Run one batch of input through model
           and return training loss on outputs

        Parameters
        ----------
        idx : int
            index of batch within train_loader
        sample : dict
            data dictionary holding one batch

        Returns
        -------
        loss: float | Tensor
            float value of training loss
        """

        self.optimizer.zero_grad(set_to_none=True)
        if self.regularizer:
            self.regularizer.reset()
        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)
        else:
            # load data to device if no preprocessor exists
            sample = {
                k: v.to(self.device)
                for k, v in sample.items()
                if torch.is_tensor(v)
            }

        if isinstance(sample["y"], torch.Tensor):
            self.n_samples += sample["y"].shape[0]
        else:
            self.n_samples += 1

        if self.mixed_precision:
            with torch.autocast(device_type=self.autocast_device_type):
                out = self.model(**sample)
        else:
            out = self.model(**sample)
        
        if self.epoch == 0 and idx == 0 and self.verbose and isinstance(out, torch.Tensor):
            print(f"Raw outputs of shape {out.shape}")

        if self.data_processor is not None:
            out, sample = self.data_processor.postprocess(out, sample)

        loss = 0.0
        #print(f"Using training loss: {training_loss.__class__.__name__}")
        
        # Prepare kwargs for loss
        loss_kwargs = {
            "y_pred": out,
            "y": sample.get("y", None),
            "x": sample.get("x", None)
        }

        # Compute weighted loss
        try:
            if self.mixed_precision:
                with torch.autocast(device_type=self.autocast_device_type):
                    loss = training_loss(**loss_kwargs)
            else:
                loss = training_loss(**loss_kwargs)
        except TypeError:
            # fallback: pass raw sample in case loss ignores query points
            if self.mixed_precision:
                with torch.autocast(device_type=self.autocast_device_type):
                    loss = training_loss(**sample, y_pred=out)
            else:
                loss = training_loss(**sample, y_pred=out)
   
        if self.regularizer:
            loss += self.regularizer.loss
        
        return loss
    
    def eval_one_batch(self,
                       sample: dict,
                       eval_losses: dict,
                       return_output: bool=False):
        """eval_one_batch runs inference on one batch
        and returns eval_losses for that batch.

        Parameters
        ----------
        sample : dict
            data batch dictionary
        eval_losses : dict
            dictionary of named eval metrics
        return_outputs : bool
            whether to return model outputs for plotting
            by default False
        Returns
        -------
        eval_step_losses : dict
            keyed "loss_name": step_loss_value for each loss name
        outputs: torch.Tensor | None
            optionally returns batch outputs
        """
        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)
        else:
            # load data to device if no preprocessor exists
            sample = {
                k: v.to(self.device)
                for k, v in sample.items()
                if torch.is_tensor(v)
            }

        self.n_samples += sample["y"].size(0)

        out = self.model(**sample)
        
        # Prepare kwargs for evaluation losses
        loss_kwargs = {
            "y_pred": out,
            "y": sample.get("y", None),
            "x": sample.get("x", None)
        }

        eval_step_losses = {}
        for loss_name, loss in eval_losses.items():
            try:
                # Filter kwargs per loss signature
                sig = inspect.signature(loss.__call__)
                filtered_kwargs = {k: v for k, v in loss_kwargs.items() if k in sig.parameters}
                eval_step_losses[loss_name] = loss(**filtered_kwargs)
            except TypeError:
                # fallback for losses that ignore query points
                filtered_kwargs = {k: v for k, v in sample.items() if k in sig.parameters}
                filtered_kwargs["y_pred"] = out
                eval_step_losses[loss_name] = loss(**filtered_kwargs)

        if return_output:
            return eval_step_losses, out
        else:
            return eval_step_losses, None

    def log_training(self, 
            epoch:int,
            time: float,
            avg_loss: float,
            train_err: float,
            avg_lasso_loss: float=None,
            lr: float=None
            ):
        """Basic method to log results
        from a single training epoch. 
        

        Parameters
        ----------
        epoch: int
        time: float
            training time of epoch
        avg_loss: float
            average train_err per individual sample
        train_err: float
            train error for entire epoch
        avg_lasso_loss: float
            average lasso loss from regularizer, optional
        lr: float
            learning rate at current epoch
        """
        # accumulate info to log to wandb
        if self.wandb_log:
            values_to_log = dict(
                train_err=train_err,
                time=time,
                avg_loss=avg_loss,
                avg_lasso_loss=avg_lasso_loss,
                lr=lr)

        msg = f"[{epoch}] time={time:.2f}, "
        msg += f"avg_loss={avg_loss:.4f}, "
        msg += f"train_err={train_err:.4f}"
        if avg_lasso_loss is not None:
            msg += f", avg_lasso={avg_lasso_loss:.4f}"

        print(msg)
        sys.stdout.flush()
        
        if self.wandb_log:
            wandb.log(data=values_to_log,
                      step=epoch+1,
                      commit=False)
    
    def log_eval(self,
                 epoch: int,
                 eval_metrics: dict):
        """log_eval logs outputs from evaluation
        on all test loaders to stdout and wandb

        Parameters
        ----------
        epoch : int
            current training epoch
        eval_metrics : dict
            metrics collected during evaluation
            keyed f"{test_loader_name}_{metric}" for each test_loader
       
        """
        values_to_log = {}
        msg = ""
        for metric, value in eval_metrics.items():
            if isinstance(value, float) or isinstance(value, torch.Tensor):
                msg += f"{metric}={value:.4f}, "
            if self.wandb_log:
                values_to_log[metric] = value       
        
        msg = f"Eval: " + msg[:-2] # cut off last comma+space
        print(msg)
        sys.stdout.flush()

        if self.wandb_log:
            wandb.log(data=values_to_log,
                      step=epoch+1,
                      commit=True)
            
        

    def resume_state_from_dir(self, save_dir):
        """
        Resume training from save_dir created by `neuralop.training.save_training_state`
        
        Params
        ------
        save_dir: Union[str, Path]
            directory in which training state is saved
            (see neuralop.training.training_state)
        """
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)

        # check for save model exists
        if (save_dir / "best_model_state_dict.pt").exists():
            save_name = "best_model"
        elif (save_dir / "model_state_dict.pt").exists():
            save_name = "model"
        else:
            raise FileNotFoundError("Error: resume_from_dir expects a model\
                                        state dict named model.pt or best_model.pt.")
        # returns model, loads other modules if provided
        self.model, self.optimizer, self.scheduler, self.regularizer, resume_epoch =\
            load_training_state(save_dir=save_dir, save_name=save_name,
                                                model=self.model,
                                                optimizer=self.optimizer,
                                                regularizer=self.regularizer,
                                                scheduler=self.scheduler)

        if resume_epoch is not None:
            if resume_epoch > self.start_epoch:
                self.start_epoch = resume_epoch
                if self.verbose:
                    print(f"Trainer resuming from epoch {resume_epoch}")


    def checkpoint(self, save_dir):
        """checkpoint saves current training state
        to a directory for resuming later. Only saves 
        training state on the first GPU. 
        See neuralop.training.training_state

        Parameters
        ----------
        save_dir : str | Path
            directory in which to save training state
        """
        if comm.get_local_rank() == 0:
            if self.save_best is not None:
                save_name = 'best_model'
            else:
                save_name = "model"
            save_training_state(save_dir=save_dir, 
                                save_name=save_name,
                                model=self.model,
                                optimizer=self.optimizer,
                                scheduler=self.scheduler,
                                regularizer=self.regularizer,
                                epoch=self.epoch
                                )
            if self.verbose:
                print(f"[Rank 0]: saved training state to {save_dir}")

    def generate_coords(self, x, y, x_range=None, y_range=None):
        """
        Generate spatiotemporal coordinates (t, x, y) for input and output tensors.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, 2, H, W] at time t=0.
        y : torch.Tensor
            Ground truth output tensor of shape [B, 2, H, W] at time t= t + delta_t.
        delta_t : float (from self)
            Time step between x and y_pred, set during class initialization.
        x_range : tuple or None
            Range of x-coordinates (left to right). Defaults to (0, W - 1).
        y_range : tuple or None
            Range of y-coordinates (top to bottom). Defaults to (0, H - 1).

        Returns
        -------
        coords : torch.Tensor
            Coordinate tensor of shape [2*B, H, W, 3] with (t, x, y), requires_grad=True.
        """
        B, C, H, W = x.shape
        device = x.device

        # Use index-based range if not provided
        if x_range is None:
            x_range = (0.0, float(W - 1))
        if y_range is None:
            y_range = (0.0, float(H - 1))

        # Spatial coordinates
        x_lin = torch.linspace(x_range[0], x_range[1], W, device=device)
        y_lin = torch.linspace(y_range[0], y_range[1], H, device=device)
        yy, xx = torch.meshgrid(y_lin, x_lin, indexing='ij')  # Shape [H, W]

        # Temporal coordinates: 0 for x, delta_t for y_pred
        t_0 = torch.full_like(xx, fill_value=0.0)
        t_dt = torch.full_like(xx, fill_value=self.delta_t)

        # Stack for each batch sample
        coords = []
        for _ in range(B):
            coords_x = torch.stack([t_0, xx, yy], dim=-1)   # [H, W, 3]
            coords_y = torch.stack([t_dt, xx, yy], dim=-1)  # [H, W, 3]
            coords.append(coords_x)
            coords.append(coords_y)

        # Stack the coordinates 
        coords = torch.stack(coords, dim=0)                
        coords.requires_grad_(True)

        return coords
    
    def sample_query_points(self, B, H, W, n_interior=2048, n_boundary=512, 
                            Lx=10.0, Ly=10.0):
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
        x_int = torch.rand(B, n_interior, 1, device=self.device) * Lx
        y_int = torch.rand(B, n_interior, 1, device=self.device) * Ly
        interior_xy = torch.cat([x_int, y_int], dim=-1)

        # normalize to [-1, 1]
        interior_norm = interior_xy.clone()
        interior_norm[..., 0] = 2.0 * interior_xy[..., 0] / Lx - 1.0
        interior_norm[..., 1] = 2.0 * interior_xy[..., 1] / Ly - 1.0

        # ----- Boundary points -----
        y_b = torch.rand(B, n_boundary, 1, device=self.device) * Ly
        x_b = torch.rand(B, n_boundary, 1, device=self.device) * Lx

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
