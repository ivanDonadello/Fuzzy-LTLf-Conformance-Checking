import torch
import pdb
from FLTLf.parser import LTLfParser
from FLTLf.converter import Converter


# TODOs
# mettere sotto GNU
# Vogliamo implementare il <-> ?
predicate_names = ["cobot_holds", "human_holds", "human_glues", "qc"]

# Each trace is a list of events, each event is a list of predicate values. Use the value 0 for predicates that were not logged in an event.
traces = []
traces.append( [[0.11, 0.12, 0.13, 0.14], [0.21, 0.22, 0.23, 0.24], [0.31, 0.32,  0.33, 0.34,], [0.41, 0.42, 0.43, 0.44]] ) 
traces.append( [[0.8, 0.3, 0.1, 0.9], [0.9, 0.9, 0.93, 0.94], [0.91, 0.03, 0.4, 0.72]] )
traces.append( [[0.84, 0.3, 0.1, 0.23], [0.9, 0.07, 0.4, 0.14], [0.21, 0.93, 0.04, 0.82], [0.51, 0.03, 0.32, 0.72], [0.63, 0.03, 0.41, 0.92], [0.12, 0.63, 0.03, 0.41]] )
traces.append( [[1, 1, 1, 1]] )
traces.append( [[0, 0, 0, 0], [0.51, 0.32, 0.89, 0.72]] )

converter: Converter = Converter(predicate_names, traces)
tensor_log: torch.Tensor = converter.log2tensor(verbose=False) # tensor_log.shape = (num_traces, num_events, num_predicates
max_t = converter.max_t
batch_size = converter.batch_size


#string formula. use !, =>, &, | and temporal operators:
# always: G
# eventually: F
# next: X
# weak next: wX
# until: U
# weak until: W
# release: R


string = "G(a -> X(b))"
string = "G((cobot_holds && human_holds) <-> X(human_glues))"
#string = "aa"
#string = "a U b"
#string = "X(X(!b))"
#string = "G((a & b) => X(c))" 
#string = "(aa W (wX cd))"
#string = "(!(wX cd)) & aa"
#string = "!wX aa"
#string = "F(b W d)" 
string = "human_glues R cobot_holds"
string = "human_glues M cobot_holds"

# Parsing into a formula
parser = LTLfParser(predicate_names, tensor_log, max_t, batch_size)
formula = parser(string)

# Instant i
i = 0

print(f"Evaluation of {string} at instant {i}:")
print(formula.eval(i))