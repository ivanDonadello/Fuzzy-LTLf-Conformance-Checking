import torch
import input
import torch.nn.functional as F
from torch.masked import masked_tensor
from itertools import chain, combinations


# tensor_log.shape = (num_traces, num_events, num_predicates)
tensor_log: torch.Tensor 
batch_size: int
maxlength: int
debug: bool

##### Node (hierarchical formulas) and tree visitor #####

class Node:
    def visit(self, i, keepdim) -> torch.tensor:
        pass
        
class Visitor:

    # visits a node (so typically in DFS)
    def visit(self, node: Node, i:int):
        
        self.i = i
        node.visit(self, i, False)
        #self.result = self.data

    # prints a debug output if debug is active, then returns the evaluation
    def retResult(self, node: Node):
        if(debug):
            print(f"{node.print()}:\n{self.data}")
        return self.data

##### propositional #####

class Predicate(Node):
    # atomic predicate
    def __init__(self, predicate_name: str):
        self.predicate_name = predicate_name

    def visit(self, v: Visitor, i:int, keepdim) -> torch.tensor:
        # set explicitly v.data
        # needs to clone the tensor for each leaf of the tree. 
        # can we do better (graph vs tree, etc)? TODO
        if(keepdim):
            v.data = F.pad(tensor_log[:, i:, input.predicate_names.index(self.predicate_name)], (0, i), mode="constant", value=torch.nan)
        else:
            v.data = (tensor_log[:, i, input.predicate_names.index(self.predicate_name)])
        
        return v.retResult(self)
        
    #old (same as in all classes). Remove TODO
    def eval(self, i: int) -> torch.tensor:
        assert i <= maxlength, f"i exceeds maxlength ({i}>{maxlength})"
        return tensor_log[:, i, input.predicate_names.index(self.predicate_name)]; 

    def print(self):
        return self.predicate_name
        
class ComparisonTerm(Node):
    # comparison term
    def __init__(self, a, b, op):
        self.a = a
        self.b = b
        self.op = op
        self.name = "comparison"

    def visit(self, v: Visitor, i:int, keepdim) -> torch.tensor:
        
        # this creates a new tensor. can we avoid this? TODO
        sat_a = self.a.visit(v, i, keepdim)

        if(isinstance(self.b, float)):
            if(keepdim):
                sat_b = torch.full(sat_a.shape, self.b)
            else: 
                sat_b = torch.full((1,batch_size), self.b)
        else:
            sat_b = self.b.visit(v, i, keepdim)

        #print(batch_size)
        #print(f"1--{sat_a}")
        #print(f"2--{sat_b}")

        # here, v.data is overwritten
        match self.op:
            case "<":
                v.data = torch.where(torch.isnan(sat_a) | torch.isnan(sat_b), torch.nan, torch.lt(sat_a,sat_b))
            case "<=":
                v.data = torch.where(torch.isnan(sat_a) | torch.isnan(sat_b), torch.nan, torch.le(sat_a,sat_b))
            case ">":
                v.data =  torch.where(torch.isnan(sat_a) | torch.isnan(sat_b), torch.nan, torch.gt(sat_a,sat_b))
            case ">=":
                v.data =  torch.where(torch.isnan(sat_a) | torch.isnan(sat_b), torch.nan, torch.ge(sat_a,sat_b))
            case "==":
                v.data =  torch.where(torch.isnan(sat_a) | torch.isnan(sat_b), torch.nan, torch.eq(sat_a,sat_b))
            case "!=":
                v.data =  torch.where(torch.isnan(sat_a) | torch.isnan(sat_b), torch.nan, torch.ne(sat_a,sat_b))

        if(not(keepdim)):
            v.data = v.data[0]
        
        return v.retResult(self)


    def eval(self, i: int) -> torch.tensor:
        assert i <= maxlength, f"i exceeds maxlength ({i}>{maxlength})"

        sat_a = self.a.eval(i)

        if(isinstance(self.b, float)):
            sat_b = self.b
        else:
            sat_b = self.b.eval(i)

        # no need to check: OR isnan(sat_b): iff isnan(sat_a)
        match self.op:
            case "<":
                return torch.where(torch.isnan(sat_a), torch.nan, torch.lt(sat_a,sat_b))
            case "<=":
                return torch.where(torch.isnan(sat_a), torch.nan, torch.le(sat_a,sat_b))
            case ">":
                return torch.where(torch.isnan(sat_a), torch.nan, torch.gt(sat_a,sat_b))
            case ">=":
                return torch.where(torch.isnan(sat_a), torch.nan, torch.ge(sat_a,sat_b))
            case "==":
                return torch.where(torch.isnan(sat_a), torch.nan, torch.eq(sat_a,sat_b))
            case "!=":
                return torch.where(torch.isnan(sat_a), torch.nan, torch.ne(sat_a,sat_b))
                
    def print(self):
        if(isinstance(self.b, float)):
            return "(" + self.a.print() + " " + self.op + " " + str(self.b) + ")"
        else:
            return "(" + self.a.print() + " " + self.op + " " + self.b.print() + ")"

class NegPredicate(Node):
    # negated atomic predicate
    def __init__(self, p: Predicate):
        self.exp = p
        self.name = "NegPredicate"
        self.children = []
 
    def visit(self, v: Visitor, i:int, keepdim) -> torch.tensor:
        # here, v.data is overwritten
        v.data = torch.sub(1, self.exp.visit(v, i, keepdim))

        if(not(keepdim)):
            v.data = v.data[0]

        return v.retResult(self)

    def eval(self, i) -> torch.tensor:
        return 1.0 - self.exp.eval(i)
    
    def print(self):
        return "NOT(" + self.exp.print() + ")"
    
class BoolConst(Node):
    # boolean contant
    def __init__(self, const):
        self.const = const               
        self.name = "Boolean"
        self.children = []

    def visit(self, v: Visitor, i:int, keepdim) -> torch.tensor:
        
        # here, v.data is overwritten
        # as any type of leaf, this uses a new tensor
        if(keepdim):
            if(self.const==1):
                v.data = torch.ones((batch_size,maxlength))
            else:
                v.data = torch.zeros((batch_size,maxlength))
        else:
            if(self.const==1):
                v.data = torch.ones(1)
            else:
                v.data = torch.zeros(1)

        return v.retResult(self)

    def eval(self, i) -> torch.tensor:
        assert i <= maxlength, f"i exceeds maxlength ({i}>{maxlength})"

        if(self.const==1):
            return torch.ones(batch_size)
        else:
            return torch.zeros(batch_size)

    def print(self):
        return "TRUE" if self.const else "FALSE"  

class And(Node):
    # n-ary conjunction 
    def __init__(self, exprs: list):
        self.exprs = exprs
        self.name = "AND"

    def visit(self, v: Visitor, i:int, keepdim) -> torch.tensor:
        
        #this sets v.data
        self.exprs[0].visit(v, i, keepdim)

        #print(f"--{v.data}")
        
        for child in self.exprs[1:]:  
            v.data = torch.minimum(v.data, child.visit(v,i, keepdim))
        #print(f"--{v.data}")
        
        return v.retResult(self)

    def eval(self, i: int) -> torch.tensor:
        assert i <= maxlength, f"i exceeds maxlength ({i}>{maxlength})"
            
        sats = torch.stack([exp.eval(i) for exp in self.exprs])
        return torch.min(sats, 0).values

    def print(self):
        strings = [ exp.print() for exp in self.exprs]
        return (" " + self.name + " ").join(strings)

class Or(Node):
    # n-ary disjunction
    def __init__(self, exprs: list):
        self.exprs = exprs
        self.name = "OR"

    def visit(self, v: Visitor, i:int, keepdim) -> torch.tensor:
        
        self.exprs[0].visit(v, i, keepdim)

        for child in self.exprs[1:]:
            v.data = torch.maximum(v.data, child.visit(v, i, keepdim))

        #if(not(keepdim)):
        #    v.data = v.data[0]

        return v.retResult(self)


    def eval(self, i: int) -> torch.tensor:
        assert i <= maxlength, f"i exceeds maxlength ({i}>{maxlength})"

        sats = torch.stack([exp.eval(i) for exp in self.exprs])
        return torch.max(sats, 0).values

    def print(self):
        strings = [ exp.print() for exp in self.exprs]
        return (" " + self.name + " ").join(strings)
    

class Implication(Node):
    # implication
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.imp = Or([Negate(a), b])
        self.name = "IMPLIES"

    def visit(self, v: Visitor, i, keepdim) -> torch.tensor:
        self.imp.visit(v, i, keepdim)
        return v.data; 

    def eval(self, i: int) -> torch.tensor:
        assert i <= maxlength, f"i exceeds maxlength ({i}>{maxlength})"

        return self.imp.eval(i)

    def print(self):
        return "(" + self.a.print() + " " + self.name + " " + self.b.print() + ")"
    
#### standard LTLf Temporal Operators ####

class Next(Node):
    # next operator
    def __init__(self, exp):   
        self.exp = exp
        self.name = "NEXT"

    def visit(self, v: Visitor, i:int, keepdim) -> torch.tensor:
               
        if(not(keepdim)):
            v.data = self.exp.visit(v, i+1, keepdim)
        else:
            v.data = self.exp.visit(v, i, keepdim)
            # shift left
            v.data = torch.roll(v.data, -1) 
            print(v.data)
            # replace with nan the last elements
            v.data[:,v.data.shape[1]-1] = torch.nan #v.data[:, -1] 
            # replace nan with 0. doublecheck TODO
            v.data[ v.data.isnan() ] = 0 

        return v.retResult(self)
  
        
    def eval(self, i: int) -> torch.tensor:
        assert i <= maxlength, f"i exceeds maxlength ({i}>{maxlength})"

        if i < maxlength - 1:  
            sats = self.exp.eval(i + 1)
            sats[ sats.isnan() ] = 0
            return sats
        else:
            return torch.zeros(batch_size)

    def print(self):
        return self.name + "(" + self.exp.print() + ")"
    
class WeakNext(Node):
    # weak next operator
    def __init__(self, exp):
        self.exp = exp
        self.name = "WEAK_NEXT"

    def visit(self, v: Visitor, i:int, keepdim) -> torch.tensor:

        if(not(keepdim)):
            v.data = self.exp.visit(v, i+1, keepdim)
        else:
            v.data = self.exp.visit(v, i, keepdim)
            # shift left
            v.data = torch.roll(v.data, -1) 
            print(v.data)
            # replace with nan the last elements
            v.data[:,v.data.shape[1]-1] = torch.nan   #v.data[:, -1]
            # replace nan with 1. doublecheck TODO
            v.data[ v.data.isnan() ] = 1 

        return v.retResult(self)
        
    def eval(self, i: int) -> torch.tensor:
        assert i <= maxlength, f"i exceeds maxlength ({i}>{maxlength})"

        if i < maxlength - 1:  
            sats = self.exp.eval(i + 1)
            sats[ sats.isnan() ] = 1
            return sats
        else:
            return torch.ones(batch_size)

    def print(self):
        return self.name + "(" + self.exp.print() + ")"
    
class Always(Node):
    # always operator
    def __init__(self, exp):
        self.exp = exp
        self.name = "ALWAYS"

    def visit(self, v: Visitor,i:int, keepdim) -> torch.tensor:

        v.data = self.exp.visit(v, i, True)  #forces True      

        #v.data = choosemin(v.data) 
        for j in reversed(range(i-1, maxlength-i-1)):#range up to -1, e qui poi faccio+1
            v.data[:, j] = torch.fmin(v.data[:, j], v.data[:, j + 1])

        if(not(keepdim)):
            v.data = v.data[:, 0]

        return v.retResult(self)
       
        
    def eval(self, i: int) -> torch.tensor:
        assert i <= maxlength, f"i exceeds maxlength ({i}>{maxlength})"

        sats = torch.stack([self.exp.eval(j) for j in range(i, maxlength)], 1)
        #print(f"))))){sats}")
        return choosemin(sats)
        
    def print(self):
        return self.name + "(" + self.exp.print() + ")"
    
class Eventually(Node):
    # eventually operator
    def __init__(self, exp):
        self.exp = exp
        self.name = "EVENTUALLY"

    def visit(self, v: Visitor, i:int, keepdim) -> torch.tensor:
        v.data = self.exp.visit(v, i, True)        
        
        #v.data = choosemax(v.data) 
        for j in reversed(range(i-1, maxlength-i-1)):#range up to -1, e qui poi faccio+1
            v.data[:, j] = torch.fmax(v.data[:, j], v.data[:, j + 1])

        if(not(keepdim)):
            v.data = v.data[:, 0]

        return v.retResult(self)

    def eval(self, i: int) -> torch.tensor:
        assert i <= maxlength, f"i exceeds maxlength ({i}>{maxlength})"

        sats = torch.stack([self.exp.eval(j) for j in range(i, maxlength)], 1)        
        return choosemax(sats)
 
    def print(self):
        return self.name + "(" + self.exp.print() + ")"
    
class Release(Node):
    # release operator
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.name = "RELEASE"

    def visit(self, v: Visitor, i:int, keepdim) -> torch.tensor:
        
        sats_a = self.a.visit(v, i, True)
        sats_b = self.b.visit(v, i, True)

        v.data = torch.zeros(sats_a.shape)

        v.data[:, -1] = sats_b[:, -1] 
        
        for j in reversed(range(i-1, maxlength-i-1)):
            v.data[:, j] = torch.fmin(sats_b[:, j], torch.fmax(sats_a[:, j], v.data[:, j + 1]))
        
        if(not(keepdim)):
            v.data = v.data[:, v.i]

        return v.retResult(self)
        
    def eval(self, i: int) -> torch.tensor:
        assert i <= maxlength, f"i exceeds maxlength ({i}>{maxlength})"

        sats_a = torch.stack([self.a.eval(j) for j in range(i, maxlength)], 1)
        sats_b = torch.stack([self.b.eval(j) for j in range(i, maxlength)], 1)
        ys = torch.zeros(sats_a.shape)

        ys[:, -1] = sats_b[:, -1] 
        
        for i in reversed(range(0, ys.shape[1] - 1)):
            ys[:, i] = torch.fmin(sats_b[:, i], torch.fmax(sats_a[:, i], ys[:, i + 1]))
        return ys[:, 0]

    def print(self):
        return "(" + self.a.print() + " " + self.name + " " + self.b.print() + ")"
    
class StrongRelease(Node):
    # strong release opeator
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.name = "STRONG_RELEASE"

    def visit(self, v: Visitor,i:int,  keepdim) -> torch.tensor:
        
        sats_a = self.a.visit(v, i, True)
        sats_b = self.b.visit(v, i, True)

        v.data = torch.zeros(sats_a.shape)
        v.data[:, -1] = torch.fmin(sats_a[:, -1], sats_b[:, -1])  
        
        for j in reversed(range(i-1, maxlength-i-1)):
            v.data[:, j] = torch.fmin(sats_b[:, j], torch.fmax(sats_a[:, j], v.data[:, j + 1]))
            
        if(not(keepdim)):
            v.data = v.data[:, v.i]

        return v.retResult(self)
        
    def eval(self, i: int) -> torch.tensor:
        assert i <= maxlength, f"i exceeds maxlength ({i}>{maxlength})"

        sats_a = torch.stack([self.a.eval(j) for j in range(i, maxlength)], 1)
        sats_b = torch.stack([self.b.eval(j) for j in range(i, maxlength)], 1)
        ys = torch.zeros(sats_a.shape)

        ys[:, -1] = torch.fmin(sats_a[:, -1], sats_b[:, -1])  
        
        for i in reversed(range(0, ys.shape[1] - 1)):
            ys[:, i] = torch.fmin(sats_b[:, i], torch.fmax(sats_a[:, i], ys[:, i + 1]))
        return ys[:, 0]

    def print(self):
        return "(" + self.a.print() + " " + self.name + " " + self.b.print() + ")"
    
class WeakUntil(Node):
    # weak until operator
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.name = "WEAK_UNTIL"

    def visit(self, v: Visitor, i:int, keepdim) -> torch.tensor:
        
        sats_a = self.a.visit(v, i, True)
        sats_b = self.b.visit(v, i, True)

        v.data = torch.zeros(sats_a.shape)
        v.data[:, -1] = torch.fmax(sats_a[:, -1], sats_b[:, -1])
                               
        for i in reversed(range(0, v.data.shape[1]-1)):
            v.data[:, i] = torch.fmax(sats_b[:, i], torch.fmin(sats_a[:, i], v.data[:, i + 1]))

        if(not(keepdim)):
            v.data = v.data[:, v.i]

        return v.retResult(self)

    def eval(self, i: int) -> torch.tensor:
        assert i <= maxlength, f"i exceeds maxlength ({i}>{maxlength})"

        sats_a = torch.stack([self.a.eval(j) for j in range(i, maxlength)], 1)
        sats_b = torch.stack([self.b.eval(j) for j in range(i, maxlength)], 1)

        ys = torch.zeros(sats_a.shape)
        ys[:, -1] = torch.fmax(sats_a[:, -1], sats_b[:, -1])

        for i in reversed(range(0, ys.shape[1] - 1)):
            ys[:, i] = torch.fmax(sats_b[:, i], torch.fmin(sats_a[:, i], ys[:, i + 1]))
        return ys[:, 0]
 

    def print(self):
        return "(" + self.a.print() + " " + self.name + " " + self.b.print() + ")"

class Until(Node):
    # until operator
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.name = "UNTIL"

    def visit(self, v: Visitor,i:int,  keepdim) -> torch.tensor:
        
        #this duplicates the tensor. can we do better? TODO
        sats_a = self.a.visit(v, i, True)
        sats_b = self.b.visit(v, i,  True)

        v.data = torch.zeros(sats_a.shape)
        v.data[:, -1] = sats_b[:, -1]
                               
        for i in reversed(range(0, v.data.shape[1]-1)):
            v.data[:, i] = torch.fmax(sats_b[:, i], torch.fmin(sats_a[:, i], v.data[:, i + 1]))

        if(not(keepdim)):
            v.data = v.data[:, v.i]

        return v.retResult(self)
 

    def eval(self, i: int) -> torch.tensor:
        assert i <= maxlength, f"i exceeds maxlength ({i}>{maxlength})"

        sats_a = torch.stack([self.a.eval(j) for j in range(i, maxlength)], 1)
        sats_b = torch.stack([self.b.eval(j) for j in range(i, maxlength)], 1)

        ys = torch.zeros(sats_a.shape)
        ys[:, -1] = sats_b[:, -1]
                               
        for i in reversed(range(0, ys.shape[1]-1)):
            ys[:, i] = torch.fmax(sats_b[:, i], torch.fmin(sats_a[:, i], ys[:, i + 1]))
        return ys[:, 0]

    def print(self):
        return "(" + self.a.print() + " " + self.name + " " + self.b.print() + ")"
    




#### Fuzzy-time LTL Temporal Operators ####
#### ALL ARE TODO (still the old version)

class Soon:
    # soon operator
    def __init__(self, exp):
        self.exp = exp
        self.name = "SOON"

    def eval(self, i: int) -> torch.tensor:
        assert i <= maxlength, f"i exceeds maxlength ({i}>{maxlength})"

        # diffently from the semantics, we do not evaluate X phi but phi
        # we retrict to instants in [i+1, min(i+eta,last)]
        sats = torch.stack([( torch.mul( self.exp.eval(j), input.weights(j-i) )) for j in range(i+1, min(i+input.eta,maxlength) )], 1)
        return choosemax(sats)
   
    def print(self):
        return self.name + "(" + self.exp.print() + ")"
    
class BoundedEventually:
    # bounded version of the eventually operator
    def __init__(self, exp, t: int):
        self.exp = exp
        self.t = t
        
        self.name = "EVENTUALLY_IN_"

    def eval(self, i: int) -> torch.tensor:
        assert i <= maxlength, f"i exceeds maxlength ({i}>{maxlength})"

        # from i to min(i+t,last)
        sats = torch.stack([ self.exp.eval(j) for j in range(i, min(i+self.t+1,maxlength) )], 1)
        return choosemax(sats)
   
    def print(self):
        return self.name +  str(self.t) + "(" + self.exp.print() + ")"
    
class BoundedGlobally:
    # bounded version of the awalys operator
    def __init__(self, exp, t: int):
        self.exp = exp
        self.t = t
        self.name = "ALWAYS_IN_"

    def eval(self, i: int) -> torch.tensor:
        assert i <= maxlength, f"i exceeds maxlength ({i}>{maxlength})"

        #from i to min(i+t,last)
        sats = torch.stack([ self.exp.eval(j) for j in range(i, min(i+self.t+1,maxlength) )], 1)
        return choosemin(sats)
   
    def print(self):
        return self.name +  str(self.t) + "(" + self.exp.print() + ")"

class Within:
    # within operator
    def __init__(self, exp, t: int):
        self.exp = exp
        self.t = t        
        self.name = "WITHIN_"

    def eval(self, i: int) -> torch.tensor:
        # we retrict to instants in [i, min(i+t+eta,last)]
        sats = torch.stack([( torch.mul( self.exp.eval(j), input.weights(j-i-self.t) )) for j in range(i, min(i+self.t+input.eta,maxlength) )], 1)
        return choosemax(sats)
   
    def print(self):
        return self.name +  str(self.t) + "(" + self.exp.print() + ")"

class BoundedUntil:
    # bounded version of the until operator
    def __init__(self, a, b, t: int):
        self.a = a
        self.b = b
        self.t = t
        self.name = "UNTIL_"

    def eval(self, i: int) -> torch.tensor:
        assert i <= maxlength, f"i exceeds maxlength ({i}>{maxlength})"

        if(self.t<1):
            return self.b.eval(i)
        else:
            # from i to min(i+t,last)
            sats_a = torch.stack([self.a.eval(j) for j in range(i, min(i+self.t+1, maxlength))], 1)
            sats_b = torch.stack([self.b.eval(j) for j in range(i, min(i+self.t+1, maxlength))], 1)

            ys = torch.zeros(sats_a.shape)
            ys[:, -1] = sats_b[:, -1]
                                
            for i in reversed(range(0, ys.shape[1]-1)):
                ys[:, i] = torch.fmax(sats_b[:, i], torch.fmin(sats_a[:, i], ys[:, i + 1]))
            return ys[:, 0]
   
    def print(self):
        return "(" + self.a.print() + " " + self.name + str(self.t) + " " + self.b.print() + ")"           


class AlmostAlways:
    # almost always operator
    def __init__(self, exp):
        self.exp = exp
        self.name = "ALMOST_ALWAYS"
        
    def eval(self, i: int) -> torch.tensor:
        
        globalmax = torch.zeros(batch_size)

        global tensor_log
        
        #this would be be exponential: 
        # try filtering each element of the powerset of indexes, of length up to eta-1
        #for exclude in list(powerset(range(i,maxlength),input.eta)):

        #instead, we use sorting 
        #to exclude incrementally i,i+1,... up to min(eta-1,last)

        #first, we evaluate without filtering any index
        sats = torch.stack([  self.exp.eval(j) for j in range(i,maxlength)], 1)
        globalmax = choosemin(sats)

        #cloning, since this could be a subcall
        orig_tensor_log = tensor_log.clone()
        tensor_log,indexes = tensor_log.sort(dim=1)
        
        exclude = []
        
        #we can exclude up to eta-1 events, so from i to min(i+eta-1,last)
        for x in range(i, min(i+input.eta,maxlength)):
            exclude.append(x)
            
            if(len(exclude)<maxlength):

                #incrementally, we exclude instants at the beginning of the sorted sequence
                keep = (x for x in range(i, maxlength) if x not in exclude)
                sats = torch.stack([  self.exp.eval(j) for j in keep], 1)

                #print(f"--- exclude {exclude} weight is {input.weights(len(exclude))}")
                #print(f"sats: {sats}")

                mins = torch.mul( choosemin(sats) , input.weights(len(exclude)))

                #print(f"mins: {mins}")

                globalmax = torch.fmax(globalmax,mins)

        #restoring the log
        tensor_log = orig_tensor_log
        return globalmax
        
    def print(self):
        return self.name + "(" + self.exp.print() + ")"
    

class BoundedAlmostAlways:
    # bounded version of the almost always operator
    def __init__(self, exp, t:int):
        self.exp = exp
        self.t = t
        self.name = "ALMOST_ALWAYS_"
        
    def eval(self, i: int) -> torch.tensor:
        
        globalmax = torch.zeros(batch_size)

        global tensor_log
        
        sats = torch.stack([  self.exp.eval(j) for j in range(i,min(i+input.eta,maxlength))], 1)
        globalmax = choosemin(sats)

        orig_tensor_log = tensor_log.clone()
        tensor_log,indexes = tensor_log.sort(dim=1)
        
        exclude = []

        # t used here
        #we can exclude up to eta-1 events, so from i to min(eta-1,i+t,last)
        for x in range(i, min(input.eta,i+self.t+1,maxlength)):
            exclude.append(x)
            print(i)
            print(str(input.eta))
            print(str(i+self.t+1))
            print(maxlength)
            
            if(i+len(exclude)<maxlength):

                print(f"--- exclude {exclude} weight is {input.weights(len(exclude))}")
                
                keep = (x for x in range(i, min(i+self.t+1,maxlength)) if x not in exclude)
                sats = torch.stack([  self.exp.eval(j) for j in keep], 1)

                print(f"sats: {sats}")

                mins = torch.mul( choosemin(sats) , input.weights(len(exclude)))

                #print(f"mins: {mins}")

                globalmax = torch.fmax(globalmax,mins)

        #restoring the log
        tensor_log = orig_tensor_log
        return globalmax
        
    def print(self):
        return self.name + str(self.t) + "(" + self.exp.print() + ")"
    


class AlmostUntil:
    # almost until operator
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.name = "ALMOST_UNTIL"
        
    def eval(self, i: int) -> torch.tensor:
        
        globalmax = torch.zeros(batch_size)

        global tensor_log
            
        sats_a = torch.stack([self.a.eval(j) for j in range(i,maxlength)], 1)
        sats_b = torch.stack([self.b.eval(j) for j in range(i,maxlength)], 1)
        
        ys = torch.zeros(sats_a.shape)
        ys[:, -1] = sats_b[:, -1]
        
        for i in reversed(range(0, ys.shape[1]-1)):
            ys[:, i] = torch.fmax(sats_b[:, i], torch.fmin(sats_a[:, i], ys[:, i + 1]))
                    
        globalmax = ys[:, 0]
        
        orig_tensor_log = tensor_log.clone()
        tensor_log,indexes = tensor_log.sort(dim=1)
            
        exclude = []

        #we can exclude up to eta-1 events, so from i to min(i+eta-1,last)
        for x in range(i, min(i+input.eta,maxlength)):
            exclude.append(x)
                
            if(i+len(exclude)<maxlength):
                
                #here I need to store it with list(), cause generators are not rewund
                keep = list(x for x in range(i, maxlength) if x not in exclude)
            
                sats_a = torch.stack([self.a.eval(j) for j in keep], 1)
                sats_b = torch.stack([self.b.eval(j) for j in keep], 1)

                ys = torch.zeros(sats_a.shape)
                ys[:, -1] = sats_b[:, -1]
                                        
                for i in reversed(range(0, ys.shape[1]-1)):
                    ys[:, i] = torch.fmax(sats_b[:, i], torch.fmin(sats_a[:, i], ys[:, i + 1]))
                    
                globalmax = torch.fmax(globalmax,ys[:,0])

        #restoring the log
        tensor_log = orig_tensor_log
        return globalmax
    
        
    def print(self):
        return "(" + self.a.print() + " " + self.name + " " + self.b.print() + ")"


#TODO BoundedAlmostUntil

class Lasts:
    # lasts operator
    def __init__(self, exp, t:int):
        self.exp = exp
        self.t = t
        self.name = "LASTS_"

    def eval(self, i: int) -> torch.tensor:
        assert i <= maxlength, f"i exceeds maxlength ({i}>{maxlength})"

        from FLTLf.parser import LTLfParser
        parser = LTLfParser()
        
        globalmax = torch.zeros(batch_size)

        #from 0 to min(t,eta-1,last)
        for j in range(0, min(self.t+1, i+input.eta, maxlength)):

            #this is the only operator when we rewrite the formula (apart from ->)
            #note: we saw the parsing takes time
            pyformula = parser("BG[" + str(self.t-j) + "](" + self.exp.print() + ")")  
            #print(f"evaluate {pyformula.print()} in instant {i}")   
            globalmax = torch.fmax(globalmax, torch.mul( pyformula.eval(i), input.weights(j) ) )

        return globalmax
   
    def print(self):
        return self.name + str(self.t) + "(" + self.exp.print() + ")"


#### Negation ####

class Negate:
    # negated formula, limited to the atomic formulae (excluding )
    def __init__(self, exp):
        self.exp = exp
        self.name = "Negation"

        if isinstance(self.exp, Negate):
            self.neg = self.exp.exp
        elif isinstance(self.exp, And):
            neg_exprs = [Negate(e) for e in self.exp.exprs]
            self.neg = Or(neg_exprs)
        elif isinstance(self.exp, Or):
            neg_exprs = [Negate(e) for e in self.exp.exprs]
            self.neg = And(neg_exprs)
        elif isinstance(self.exp, Implication):
            self.neg = And([self.exp.a, Negate(self.exp.b)])
        elif isinstance(self.exp, BoolConst):
            self.neg = BoolConst(1.0 - self.exp.x)
        elif isinstance(self.exp, Next):
            self.neg = Next(Negate(self.exp.exp))
        elif isinstance(self.exp, WeakNext):
            self.neg = WeakNext(Negate(self.exp.exp))
        elif isinstance(self.exp, Eventually):
            self.neg = Always(Negate(self.exp.exp))
        elif isinstance(self.exp, Always):
            self.neg = Eventually(Negate(self.exp.exp))
        elif isinstance(self.exp, Predicate):
            self.neg = NegPredicate(self.exp)
        elif isinstance(self.exp, Until):
            self.neg = Release(Negate(self.exp.a), Negate(self.exp.b))
        elif isinstance(self.exp, WeakUntil):
            self.neg = StrongRelease(Negate(self.exp.a), Negate(self.exp.b))
        elif isinstance(self.exp, Release):
            self.neg = Until(Negate(self.exp.a), Negate(self.exp.b))
        elif isinstance(self.exp, StrongRelease):
            self.neg = WeakUntil(Negate(self.exp.a), Negate(self.exp.b))
        else:
            raise Exception(f"Negation not implemented for {self.exp.name}.") 

    def visit(self, v: Visitor,i:int,  keepdim):
        self.neg.visit(v, i, keepdim)
        #self.data = self.neg.data  
        return v.data


    def eval(self, i: int) -> torch.tensor:
        return self.neg.eval(i)  
    
    def print(self):
        return self.neg.print()
        


#### Aux ####

#as torch.min, but returns nan iff only nans are present 
def choosemin(tensor):

    #only works for simple formulas. to investigate
    #masked = masked_tensor(tensor, ~torch.isnan(tensor))
    #return torch.amin(masked, 1)

    tensor = torch.nan_to_num(tensor, nan=2) 
    tmp = torch.min(tensor, 1).values
    tmp[tmp == 2] = torch.nan
    return tmp

#same, for max
def choosemax(tensor):
    tensor = torch.nan_to_num(tensor, nan=-2)
    tmp = torch.max(tensor, 1).values
    tmp[tmp == -2] = torch.nan
    return tmp

def powerset(alist,maxlen):
    s = list(alist)
    return chain.from_iterable(combinations(s, r) for r in range(min(maxlen,len(s))+1))

# generates a set of subtensors, for instants in {i,...,length}
def estract_sub_logs(tensor_log: torch.Tensor, i:int, length:int):
    sublogs = []
    for indices in list(powerset(range(i,length))):
        if len(indices) > 0:#TODO FORSE NO
            tmp_sublog = tensor_log[:, indices, :]
            sublogs.append(tmp_sublog)
    return sublogs


