from functools import partialmethod
from pathlib import Path
from typing import List, Union, Optional
import torch
import random 
import os

from .tensor_dataset import TensorDataset
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
#from data_transform import inspect_data 

class PTDataset:
    """PTDataset is a base Dataset class based on the dataloader from the NeuralOps Library, adpated for out PDE problem. 
            PTDatasets contain input-output pairs a(x), u(x) and may also
            contain additional information, e.g. function parameters,
            input geometry or output query points.

            Datasets are indexed by their spatial resolution and a 'conductivity multiplier' value that can be used to distinguish between simulations with different D values. 

        All datasets are required to expose the following attributes after init:

        train_db: torch.utils.data.Dataset of training examples
        test_db:  ""                       of test examples
        data_processor: neuralop.data.transforms.DataProcessor to process data examples
            optional, default is None

        """
    def __init__(self,
                 root_dir: Union[Path, str],
                 dataset_name: str,
                 n_train: int,
                 n_tests: List[int],
                 cm_train: float,
                 cm_tests: List[float],
                 batch_size: int,
                 test_batch_sizes: List[int],
                 train_resolution: int,
                 test_resolutions: List[int],
                 encode_input: bool=False, 
                 encode_output: bool=True, 
                 encoding="channel-wise",
                 input_subsampling_rate=None,
                 output_subsampling_rate=None,
                 channel_dim=1,
                 channels_squeezed=False,
                    ):
        

        """PTDataset

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
        cm_train: Float
            Identifier for the conductivity scaling for the training tensor dataset
        cm_tests: List[float]
            Identifiers for the conductivity scaling for the testing tensor datasets
        batch_size : int
            batch size of training set
        test_batch_sizes : List[int]
            batch size of test sets
        train_resolution : int
            resolution of data for training set
        test_resolutions : List[int], optional
            resolution of data for testing sets, by default [16,32]
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
            Only applies when there is only one data channel, as in our example problems
            Defaults to False to capture mutli-channel input
        """
        
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        
        self.root_dir = root_dir

        # save dataloader properties for later
        self.train_resolution = train_resolution
        self.batch_size = batch_size
        self.test_resolutions = test_resolutions
        self.test_batch_sizes = test_batch_sizes
        self.cm_train = cm_train
        self.cm_tests = cm_tests
            
        # Load training data or the evaluation data - based on the name of the path:
        keyword = "Full"
        if keyword in os.path.basename(root_dir):
            data = torch.load(
            Path(root_dir).joinpath(f"{dataset_name}_eval_{train_resolution}_{cm_train}.pt").as_posix()
            )
            print(
                f"Loading Evaluation Dataset for resolution {train_resolution} with {n_train} samples with {cm_train} conductivity multiplier"
            )
        else:
            data = torch.load(
            Path(root_dir).joinpath(f"{dataset_name}_train_{train_resolution}_{cm_train}.pt").as_posix()
            )
            print(
                f"Loading Training Dataset for resolution {train_resolution} with {n_train} samples with {cm_train} conductivity multiplier"
            )

        # Load the input data 
        x_train = data["x"].type(torch.float32).clone()
        if channels_squeezed:
            print('unsqueezing channels')
            x_train = x_train.unsqueeze(channel_dim)

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
        train_input_indices = [slice(0, n_train, None)] + [slice(None, None, rate) for rate in input_subsampling_rate]
        train_input_indices.insert(channel_dim, slice(None))
        x_train = x_train[train_input_indices]
        
        # Load the output data
        y_train = data["y"].clone()
        if channels_squeezed:
            print('unsqueezing channels')
            y_train = y_train.unsqueeze(channel_dim)

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
        train_output_indices = [slice(0, n_train, None)] + [slice(None, None, rate) for rate in output_subsampling_rate]
        train_output_indices.insert(channel_dim, slice(None))
        y_train = y_train[train_output_indices]
        
        del data

        # Fit optional encoders to train data
        # Actual encoding happens within DataProcessor
        if encode_input:
            if encoding == "channel-wise":
                reduce_dims = list(range(x_train.ndim))
                # preserve mean for each channel
                reduce_dims.pop(channel_dim)
            elif encoding == "pixel-wise":
                reduce_dims = [0]

            input_encoder = UnitGaussianNormalizer(dim=reduce_dims)
            input_encoder.fit(x_train)
        else:
            input_encoder = None

        if encode_output:
            if encoding == "channel-wise":
                reduce_dims = list(range(y_train.ndim))
                # preserve mean for each channel
                reduce_dims.pop(channel_dim)
            elif encoding == "pixel-wise":
                reduce_dims = [0]

            output_encoder = UnitGaussianNormalizer(dim=reduce_dims)
            output_encoder.fit(y_train)
        else:
            output_encoder = None

        # Save train dataset
        self._train_db = TensorDataset( 
            x_train,
            y_train,
        )

        # create DataProcessor
        self._data_processor = DefaultDataProcessor(in_normalizer=input_encoder,
                                                   out_normalizer=output_encoder)



        # if loading calibration datasets, skip loading the training data:
        if keyword in os.path.basename(root_dir):
            print(
                " Evaluation Set - Skipping Test Loading "
            )
        else: 
            # load test data
            self._test_dbs = {}
            for (res, n_test, cm_test) in zip(test_resolutions, n_tests, cm_tests):

                print(
                    f"Loading Testing Dataset for resolution {res} with {n_test} samples with {cm_test} conductivity multiplier"
                )
                data = torch.load(Path(root_dir).joinpath(f"{dataset_name}_test_{res}_{cm_test}.pt").as_posix())


                x_test = data["x"].type(torch.float32).clone()
                if channels_squeezed:
                    print('unsqueezing channels')
                    x_test = x_test.unsqueeze(channel_dim)
                print(f"Input Tensor: {x_test.shape}")
                # optionally subsample along data indices
                test_input_indices = [slice(0, n_test, None)] + [slice(None, None, rate) for rate in input_subsampling_rate] 
                test_input_indices.insert(channel_dim, slice(None))
                x_test = x_test[test_input_indices]
                
                y_test = data["y"].clone()
                if channels_squeezed:
                    print('unsqueezing channels')
                    y_test = y_test.unsqueeze(channel_dim)
                print(f"Output Tensor: {y_test.shape}")
                test_output_indices = [slice(0, n_test, None)] + [slice(None, None, rate) for rate in output_subsampling_rate] 
                test_output_indices.insert(channel_dim, slice(None))
                y_test = y_test[test_output_indices]

                del data

                test_db = TensorDataset(
                    x_test,
                    y_test,
                )
                self._test_dbs[(res, cm_test)] = test_db
                #self._test_dbs[res] = test_db
    
    @property
    def data_processor(self):
        return self._data_processor
    
    @property
    def train_db(self):
        return self._train_db
    
    @property
    def test_dbs(self):
        return self._test_dbs