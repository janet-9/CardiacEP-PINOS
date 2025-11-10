__version__ = '1.0.2'

from .data import load_2D_AP, load_2D_AP_eval
from .training import Trainer
from .losses import APLoss, OperatorBackboneLoss, RMSELoss, WeightedSumLoss, LpLoss, BoundaryLoss, APFFTLoss, ICLoss, BCNeumann, AdaptiveTrainingLoss