import torch
import torch.nn.functional as F
from typing import List

import input

class Converter:
    
    def __init__(self, predicate_names: List[str], logtraces: List):
        self.predicate_names = predicate_names
        self.logtraces = logtraces
        self.maxlength: int = 0 # Maximum length across traces
        self.batch_size : int = 0 # Number of traces
        
    def log2tensor(self, stringFormula: str, verbose: bool=False) -> torch.Tensor:
        
        self.maxlength = max(len(t) for t in self.logtraces)
        padded_tensors = [self.addPadding(t) for t in self.logtraces]
        tensor_log = torch.stack(padded_tensors, dim=0)

        if verbose:
            print(tensor_log)

        
        #print(tensor_log[:,:,[1,2]])

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

        tensor_log = tensor_log[:,:,usedindexes]
        input.predicate_names = used
        self.batch_size = tensor_log.shape[0]

        if verbose:
            print("new tensor log: ")
            print(tensor_log)

        return tensor_log
        
    def addPadding(self, data: List) -> torch.Tensor:
        padded_data = F.pad(torch.tensor(data, dtype=torch.float), (0, 0, 0, self.maxlength - len(data)), mode="constant", value=torch.nan)
        return padded_data