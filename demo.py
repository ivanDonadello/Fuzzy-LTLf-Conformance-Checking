from FLTLf.parser import LTLfParser
from FLTLf.converter import Converter
from FLTLf import core
import input
import torch
import traceback

verbose = False
skippadding = False
core.device = "cpu"
core.debug = False
input.i = 0

predicate_names = ["cobot_holds", "human_holds", "human_glues", "qc"]

traces = []
traces.append( [[0.11, 0.12, 0.13, 0.14], [0.21, 0.22, 0.23, 0.24], [0.31, 0.32,  0.33, 0.34,], [0.41, 0.42, 0.43, 0.44]] ) 
traces.append( [[0.08, 0.3, 0.1, 0.9], [0.9, 0.9, 0.93, 0.94], [0.91, 0.03, 0.4, 0.72]] )
traces.append( [[0.14, 0.3, 0.4, 0.23], [0.2, 0.07, 0.4, 0.14], [0.2, 0.93, 0.7, 0.82], [0.31, 0.03, 0.42, 0.72], [0.33, 0.03, 0.41, 0.92], [0.12, 0.63, 0.03, 0.41]] )
traces.append( [[0.2, 1, 0.3, 1], [0.6, 1, 0.43, 1], [0.6, 1, 0.4, 1]] )
traces.append( [[0.3, 0, 0, 0], [0.51, 0.32, 0.89, 0.72]] )

traces = torch.rand(2, 3, len(predicate_names), dtype=torch.half)

formula = "F(((G(cobot_holds > 0.1)) & F cobot_holds) > 0.6)"

converter = Converter(predicate_names)    
tensor_log = converter.log2tensor(traces, verbose, skippadding) 

core.tensor_log, input.predicate_names = converter.slice_tensor_log(tensor_log, formula, verbose)
core.tensor_log = core.tensor_log.to(core.device)

core.batch_size = converter.batch_size
core.maxlength = converter.maxlength


# Parsing into a formula
parser = LTLfParser()

try:
    pyformula = parser(formula)

    # Instant i
    i = 0
    
    visitor = core.Visitor()
    print(f"Evaluation of {pyformula.print()} at instant {i} :")
    print(visitor.visit(pyformula, i))

    
except Exception as e:
    print(traceback.format_exc())
