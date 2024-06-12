import torch
import torch.nn.functional as F
from typing import List
import pdb


class Converter:
    """A class for converting a fuzzy log from Python lists to a three-dimensional torch Tensor"""
    def __init__(self, predicate_names: List[str], traces: List):
        self.predicate_names = predicate_names
        self.traces = traces
        self.max_t: int = 0 # Maximum length across traces
        self.batch_size : int = 0 # Number of traces
        
    def OLDlog2tensor(self, verbose: bool=False) -> torch.Tensor:
        self.max_t = max(len(t[0]) for t in self.traces)
        padded_tensors = [self.OLDaddPadding(t) for t in self.traces] # The log
        tensor_log = torch.stack(padded_tensors, dim=0)
        if verbose:
            print(tensor_log)
        self.batch_size = tensor_log.shape[0]
        return tensor_log
    
    def OLDaddPadding(self, data: List) -> torch.Tensor:
        padded_data = [F.pad(torch.tensor(tensor).float(), (0, self.max_t - len(tensor)), mode="constant", value=torch.nan) for tensor in data]
        return torch.stack(padded_data, dim=1)
        
    def log2tensor(self, verbose: bool=False) -> torch.Tensor:
        self.max_t = max(len(t) for t in self.traces)
        padded_tensors = [self.addPadding(t) for t in self.traces]
        tensor_log = torch.stack(padded_tensors, dim=0)
        if verbose:
            print(tensor_log)
        self.batch_size = tensor_log.shape[0]
        return tensor_log
        
    def addPadding(self, data: List) -> torch.Tensor:
        padded_data = F.pad(torch.tensor(data, dtype=torch.float), (0, 0, 0, self.max_t - len(data)), mode="constant", value=torch.nan)
        return padded_data