from FLTLf.parser import LTLfParser
from FLTLf.converter import Converter
from FLTLf import core
import input
import traceback

# preliminary log manipulation and padding 
converter: Converter = Converter(input.predicate_names,input.traces)
# prepares the log, also slicing out predicates not in the formula 
core.tensor_log = converter.log2tensor(input.formula,verbose=False)
# number of log traces
core.batch_size = converter.batch_size
# length of longest trace
core.maxlength = converter.maxlength

# Parsing into a formula
parser = LTLfParser()

try:
    pyformula = parser(input.formula)

    # Instant i
    i = input.i

    print("")
    print(f"Evaluation of {pyformula.print()} at instant {i} :")
    print(pyformula.eval(i))
    print("")
    
except Exception as e:
    print(traceback.format_exc())