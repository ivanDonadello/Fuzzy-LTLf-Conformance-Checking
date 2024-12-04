import torch
import time
import torch.nn.functional as F
from typing import List

import input

class Converter:
    
    def __init__(self, predicate_names: List[str]):#, logtraces: List):
        self.predicate_names = predicate_names
        self.maxlength: int = 0 # Maximum length across traces
        self.batch_size : int = 0 # Number of traces
        
    def log2tensor(self, tensor_log, verbose: bool=False, skippadding=False) -> torch.Tensor:
        
        self.maxlength = max(len(t) for t in tensor_log)
        self.batch_size = len(tensor_log)

        if not(skippadding):
            padded_tensors = [self.addPadding(t) for t in tensor_log]
            tensor_log = torch.stack(padded_tensors, dim=0)
        
        return tensor_log
    
    def slice_tensor_log(self, tensor_log: torch.Tensor, stringFormula: str, verbose: bool=False):
        
        used = []
        usedindexes = []
        
        for p in self.predicate_names:
            index = self.predicate_names.index(p)
            if p in stringFormula:
                used += [p]
                usedindexes += [index]
            elif verbose:
                print(f"slicing out predicate {p} at index {index}")

        if verbose:
            print(f"remaining indexes: {usedindexes}")

        if(len(usedindexes) < len(self.predicate_names)): 
            tensor_log = tensor_log[:,:,usedindexes]
            predicate_names = used
            self.batch_size = tensor_log.shape[0]
        else:
            predicate_names = self.predicate_names

        return tensor_log, predicate_names
        
    def addPadding(self, data: List) -> torch.Tensor:
        #padded_data = F.pad(torch.tensor(data, dtype=torch.float), (0, 0, 0, self.maxlength - len(data)), mode="constant", value=torch.nan)
        #this is what the warning (for the above) suggested
        padded_data = F.pad(data.clone().detach(), (0, 0, 0, self.maxlength - len(data)), mode="constant", value=torch.nan)
        
        return padded_data
    
