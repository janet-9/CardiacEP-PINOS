from functools import partialmethod
from pathlib import Path
from typing import List, Union, Optional

import torch
import random 


from .tensor_dataset import TensorDataset
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
#from data_transform import inspect_data 

class PTDataset_eval:
    """PTDataset is a base Dataset class for our library.
            PTDatasets contain input-output pairs a(x), u(x) and may also
            contain additional information, e.g. function parameters,
            input geometry or output query points.

            datasets may implement a download flag at init, which provides
            access to a number of premade datasets for sample problems provided
            in our Zenodo archive. 

        NEW: No loading of test-train split data: 
          All datasets are required to expose the following attributes after init:

        eval_db: torch.utils.data.Dataset of evaluation examples
        data_processor: neuralop.data.transforms.DataProcessor to process data examples
            optional, default is None

            NEW: Added in the inspect data function, designed to analyse and visualise 
            the datasets loaded. 
        """
    def __init__(self,
                 root_dir: Union[Path, str],
                 dataset_name: str,
                 n_eval: int,
                 cm_eval:float,
                 batch_size: int,
                 eval_resolution: int,
                 encode_input: bool=True, 
                 encode_output: bool=True, 
                 encoding="channel-wise",
                 input_subsampling_rate=None,
                 output_subsampling_rate=None,
                 channel_dim=1,
                 channels_squeezed=False):
        

        """PTDataset

        Parameters
        ----------
        root_dir : Union[Path, str]
            root at which to download data files
        dataset_name : str
            prefix of pt data files to store/access
        n_eval : int
            number of evaluation instances
        cm_eval: float,
            signifier for the conductivity scaling for the simulation dataset
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
        channels_squeezed : bool, optional
            If the channels dim is 1, whether that is explicitly kept in the saved tensor. 
            If not, we need to unsqueeze it to explicitly have a channel dim. 
            Only applies when there is only one data channel
            Defaults to False
        """
        
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        
        self.root_dir = root_dir

        # save dataloader properties for later
        self.batch_size = batch_size
            
        # Load the data for model evaluation
        
        data = torch.load(
        Path(root_dir).joinpath(f"{dataset_name}_eval_{eval_resolution}_{cm_eval}.pt").as_posix()
        )

        x_eval = data["x"].type(torch.float32).clone()
        if channels_squeezed:
            x_eval = x_eval.unsqueeze(channel_dim)

        # optionally subsample along data indices
        ## Input subsampling 
        input_data_dims = data["x"].ndim - 2 # batch and channels
        # convert None and 0 to 1
        if not input_subsampling_rate:
            input_subsampling_rate = 1
        if not isinstance(input_subsampling_rate, list):
            # expand subsampling rate along dims if one per dim is not provided
            input_subsampling_rate = [input_subsampling_rate] * input_data_dims
        # make sure there is one subsampling rate per data dim
        assert len(input_subsampling_rate) == input_data_dims
        # Construct full indices along which to grab X
        eval_input_indices = [slice(0, n_eval, None)] + [slice(None, None, rate) for rate in input_subsampling_rate]
        eval_input_indices.insert(channel_dim, slice(None))
        x_eval = x_eval[eval_input_indices]
        
        y_eval = data["y"].clone()
        if channels_squeezed:
            y_eval = y_eval.unsqueeze(channel_dim)

        ## Output subsampling
        output_data_dims = data["y"].ndim - 2
        # convert None and 0 to 1
        if not input_subsampling_rate:
            output_subsampling_rate = 1
        if not isinstance(output_subsampling_rate, list):
            # expand subsampling rate along dims if one per dim is not provided
            output_subsampling_rate = [output_subsampling_rate] * output_data_dims
        # make sure there is one subsampling rate per data dim
        assert len(output_subsampling_rate) == output_data_dims

        # Construct full indices along which to grab Y
        eval_output_indices = [slice(0, n_eval, None)] + [slice(None, None, rate) for rate in output_subsampling_rate]
        eval_output_indices.insert(channel_dim, slice(None))
        y_eval = y_eval[eval_output_indices]
        
        del data

        # Fit optional encoders to eval data
        # Actual encoding happens within DataProcessor
        if encode_input:
            if encoding == "channel-wise":
                reduce_dims = list(range(x_eval.ndim))
                # preserve mean for each channel
                reduce_dims.pop(channel_dim)
            elif encoding == "pixel-wise":
                reduce_dims = [0]

            input_encoder = UnitGaussianNormalizer(dim=reduce_dims)
            input_encoder.fit(x_eval)
        else:
            input_encoder = None

        if encode_output:
            if encoding == "channel-wise":
                reduce_dims = list(range(y_eval.ndim))
                # preserve mean for each channel
                reduce_dims.pop(channel_dim)
            elif encoding == "pixel-wise":
                reduce_dims = [0]

            output_encoder = UnitGaussianNormalizer(dim=reduce_dims)
            output_encoder.fit(y_eval)
        else:
            output_encoder = None

        # Save eval dataset
        self._eval_db = TensorDataset( 
            x_eval,
            y_eval,
        )

        # create DataProcessor
        self._data_processor = DefaultDataProcessor(in_normalizer=input_encoder,
                                                   out_normalizer=output_encoder)
    
    @property
    def data_processor(self):
        return self._data_processor
    
    @property
    def eval_db(self):
        return self._eval_db
    
    @property
    def test_dbs(self):
        return self._test_dbs