from FLTLf.parser import LTLfParser
from FLTLf.converter import Converter
from FLTLf import core
import input
import traceback
import torch

# preliminary log manipulation and padding 
converter: Converter = Converter(input.predicate_names,input.traces)
# prepares the log, also slicing out predicates not in the formula 
#core.tensor_log = converter.log2tensor(input.formula,verbose=False)
core.tensor_log = converter.log2tensor(verbose=False) 
core.tensor_log, input.predicate_names = converter.slice_tensor_log(core.tensor_log, input.formula, verbose=False)
# number of log traces
core.batch_size = converter.batch_size
# length of longest trace
core.maxlength = converter.maxlength
# debug (see main.py)
core.debug = input.debug

# Parsing into a formula
parser = LTLfParser()

try:
    pyformula = parser(input.formula)

    # Instant i
    i = input.i

    #print("")
    #print(f"Log max legnth is {converter.maxlength}")
    print(f"Evaluation of {pyformula.print()} at instant {i} :")
    #print(pyformula.eval(i))
    #print("")

    #print("0")
    #print(core.tensor_log)
    
    visitor = core.Visitor() 
    visitor.visit(pyformula, i)

    print("Result:")
    
    #as big as the original tensor
    #print(visitor.data)
    #a 1-dimension tensor, for each trace
    print(visitor.data)
    #the old evaluation
    
    #print(core.maxlength)

    if(input.debug):
        print(f"{pyformula.eval(i)} - old code")
    

    
except Exception as e:
    print(traceback.format_exc())