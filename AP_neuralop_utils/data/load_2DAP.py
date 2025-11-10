import logging
import os
from pathlib import Path
from typing import Union, List
from torch.utils.data import DataLoader
from .pt_dataset import PTDataset
from .pt_eval_dataset import PTDataset_eval
#from neuralop.utils import get_project_root
from pathlib import Path

logger = logging.Logger(logging.root.level)

# Define the class for loading the testing and training datasets. 
class TwoDAPDataset(PTDataset):
    """
    Load the dataset for the 2D Aliev Panfilov simulation data in a train-test split
    """
    def __init__(self,
                 root_dir: Union[Path, str],
                 dataset_name:str,
                 n_train: int,
                 n_tests: List[int],
                 cm_train: float,
                 cm_tests: List[float],
                 batch_size: int,
                 test_batch_sizes: List[int],
                 train_resolution: int,
                 test_resolutions: List[int],
                 encode_input: bool=True, 
                 encode_output: bool=True, 
                 encoding="channel-wise",
                 channel_dim=1,
                 subsampling_rate=None,
                 download: bool=True, 
                 ):

        """
        Parameters
        ----------
        root_dir : Union[Path, str]
            root at which to download data files
        dataset_name : str
            prefix of pt data files to store/access
        n_train : int
            number of train instances
        n_tests : List[int]
            number of test instances per test dataset
        batch_size : int
            batch size of training set
        test_batch_sizes : List[int]
            batch size of test sets
        train_resolution : int
            resolution of data for training set
        test_resolutions : List[int], optional
            resolution of data for testing sets
        encode_input : bool, optional
            whether to normalize inputs in provided DataProcessor,
            by default False
        encode_output : bool, optional
            whether to normalize outputs in provided DataProcessor,
            by default True
        encoding : str, optional
            parameter for input/output normalization. Whether
            to normalize by channel ("channel-wise") or 
            by pixel ("pixel-wise"), default "channel-wise"
        input_subsampling_rate : int or List[int], optional
            rate at which to subsample each input dimension, by default None
        output_subsampling_rate : int or List[int], optional
            rate at which to subsample each output dimension, by default None
        channel_dim : int, optional
            dimension of saved tensors to index data channels, by default 1
        cm: Float
            Identifier for the conductivity scaling for the tensor dataset
        """

        # convert root dir to Path
        # check the root directory to see if it needs changing...
        #print(root_dir)
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        if not root_dir.exists():
            root_dir.mkdir(parents=True)
            
        # 
        super().__init__(root_dir=root_dir,
                       dataset_name=dataset_name,
                       n_train=n_train,
                       n_tests=n_tests,
                       cm_train=cm_train,
                       cm_tests=cm_tests,
                       batch_size=batch_size,
                       test_batch_sizes=test_batch_sizes,
                       train_resolution=train_resolution,
                       test_resolutions=test_resolutions,
                       encode_input=encode_input,
                       encode_output=encode_output,
                       encoding=encoding,
                       channel_dim=channel_dim,
                       input_subsampling_rate=subsampling_rate,
                       output_subsampling_rate=subsampling_rate,
                       )


#Defining the class for loading the evaluation dataset
class TwoDAPDataset_eval(PTDataset_eval):
    """
    Load the dataset for the 2D Aliev Panfilov simulation data for use as an evaluation set
    """
    def __init__(self,
                 root_dir: Union[Path, str],
                 dataset_name: str,
                 n_eval: int,
                 cm_eval: float,
                 batch_size: int,
                 eval_resolution: int,
                 encode_input: bool=True, 
                 encode_output: bool=True, 
                 encoding="channel-wise",
                 channel_dim=1,
                 subsampling_rate=None,
                 download: bool=True,
                 ):

        """
        Parameters
        ----------
        root_dir : Union[Path, str]
            root at which to download data files
        dataset_name : str
            prefix of pt data files to store/access
        n_eval : int
            number of evaluation instances
        cm_eval: float
            signifier for the conductivity scaling for the simulation of the datset
        batch_size : int
            batch size of evaluation set
        eval_resolution : int
            resolution of data for evaluation set
        encode_input : bool, optional
            whether to normalize inputs in provided DataProcessor,
            by default False
        encode_output : bool, optional
            whether to normalize outputs in provided DataProcessor,
            by default True
        encoding : str, optional
            parameter for input/output normalization. Whether
            to normalize by channel ("channel-wise") or 
            by pixel ("pixel-wise"), default "channel-wise"
        input_subsampling_rate : int or List[int], optional
            rate at which to subsample each input dimension, by default None
        output_subsampling_rate : int or List[int], optional
            rate at which to subsample each output dimension, by default None
        channel_dim : int, optional
            dimension of saved tensors to index data channels, by default 1
        cm: Float
            Identifier for the conductivity scaling for the tensor dataset
        """

        # convert root dir to Path
        # check the root directory to see if it needs changing...
        #print(root_dir)
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        if not root_dir.exists():
            root_dir.mkdir(parents=True)
            
        # 
        super().__init__(root_dir=root_dir,
                       dataset_name=dataset_name,
                       n_eval=n_eval,
                       cm_eval = cm_eval,
                       batch_size=batch_size,
                       eval_resolution=eval_resolution,
                       encode_input=encode_input,
                       encode_output=encode_output,
                       encoding=encoding,
                       channel_dim=channel_dim,
                       input_subsampling_rate=subsampling_rate,
                       output_subsampling_rate=subsampling_rate,
                      )
        

# initial test to load the 100x100 2P AP chaotic dataset from the original EP PINNs paper
# Load the local datasets: 


# NEW: adapted for the 2 channel encoding (V,W) required for the PDE loss. 
def load_2D_AP(n_train,
    n_tests,
    cm_train,
    cm_tests,
    batch_size,
    test_batch_sizes,
    data_root,
    dataset_name,
    train_resolution,
    test_resolutions,
    encode_input,
    encode_output,
    encoding="channel-wise",
    channel_dim=1,
    ):

    dataset = TwoDAPDataset(root_dir = data_root,
                           dataset_name = dataset_name,
                           n_train=n_train,
                           n_tests=n_tests,
                           cm_train=cm_train,
                           cm_tests=cm_tests,
                           batch_size=batch_size,
                           test_batch_sizes=test_batch_sizes,
                           train_resolution=train_resolution,
                           test_resolutions=test_resolutions,
                           encode_input=encode_input,
                           encode_output=encode_output,
                           channel_dim=channel_dim,
                           encoding=encoding,
                           download=True)
    
    train_loader = DataLoader(dataset.train_db,
                              batch_size=batch_size,
                              num_workers=0,
                              pin_memory=True,
                              persistent_workers=False,)
    

    
    test_loaders = {}
    for res, cm_test, test_bsize in zip(test_resolutions, cm_tests, test_batch_sizes):
        key = (res, cm_test)  
        if key not in dataset.test_dbs:
            print(f"Warning: no test dataset for res={res}, conmul={cm_test}")
            continue
        test_loaders[key] = DataLoader(
            dataset.test_dbs[key],
            batch_size=test_bsize,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
        )
        
        '''
        test_loaders[res] = DataLoader(dataset.test_dbs[res],
                                       batch_size=test_bsize,
                                       shuffle=False,
                                       num_workers=0,
                                       pin_memory=True,
                                       persistent_workers=False,)
        '''
      
    
    return train_loader, test_loaders, dataset.data_processor


# NEW: adapted for the 2 channel encoding (V,W) required for the PDE loss - this returns the evaluation dataset (no test train split)
def load_2D_AP_eval(n_eval,
    cm_eval,
    batch_size,
    eval_resolution,
    data_root,
    dataset_name,
    encode_input,
    encode_output,
    encoding="channel-wise",
    channel_dim=1):

    dataset = TwoDAPDataset_eval(root_dir = data_root,
                           dataset_name=dataset_name,
                           n_eval=n_eval,
                           cm_eval=cm_eval,
                           batch_size=batch_size,
                           eval_resolution=eval_resolution,
                           encode_input=encode_input,
                           encode_output=encode_output,
                           channel_dim=channel_dim,
                           encoding=encoding,
                           download=True)
    
    
    # return dataloaders for backwards compat
    eval_loader = DataLoader(dataset.eval_db,
                              batch_size=batch_size,
                              num_workers=0,
                              pin_memory=True,
                              persistent_workers=False,)
    
    return eval_loader, dataset.data_processor
    


    