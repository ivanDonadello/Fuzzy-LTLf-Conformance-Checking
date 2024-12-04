from FLTLf.parser import LTLfParser
from FLTLf import core
import input
import traceback
import torch
import torch.nn.functional as F

#padding, for this file only
def mainFilePadding(data, maxlength) -> torch.Tensor:
    return F.pad(torch.tensor(data, dtype=torch.float), (0, 0, 0, maxlength - len(data)), mode="constant", value=torch.nan)

core.maxlength = max(len(t) for t in input.traces)
core.batch_size = len(input.traces)

padded_tensors = [mainFilePadding(t,core.maxlength) for t in input.traces]
core.tensor_log = torch.stack(padded_tensors, dim=0)

core.debug = input.debug

print(core.tensor_log)

parser = LTLfParser()

try:
    pyformula = parser(input.formula)

    print(f"Evaluation of {pyformula.print()} at instant {input.i} :")
    
    visitor = core.Visitor() 
    visitor.visit(pyformula, input.i)

    print("====Result:====")
    
    #a 1-dim tensor, for each trace
    print(visitor.data.to(torch.float)) #dummy casting to avoid printing dtype
    
except Exception as e:
    print(traceback.format_exc())




