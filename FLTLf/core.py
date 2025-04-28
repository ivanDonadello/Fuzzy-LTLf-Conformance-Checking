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
type = torch.half #float = 4 byte, half = 2 byte


##### nodes (of the syntactic tree) and tree visitor #####

class Node:
    def visit(self, i, keepdim) -> torch.tensor:
        pass
        
class Visitor:

    # visits a node (so typically in DFS)
    def visit(self, node: Node, i:int):
        node.visit(self, i, False)

    # index i used only for debugging
    def retResult(self, node: Node, i:int ):
        if(debug):
            print(f"{node.print()} at {i}:\n{self.data}")
        return self.data


# general idea: for a node, visit() doesn't create new tensors of 
# evaluations, but the current partial evaluation of the subtree 
# is updated bottom-up from leaves to root. 
# exceptions are, apart from leaves (predicates, constants), 
# the comparison nodes, the and/or nodes


##### propositional #####

class Predicate(Node):
    # atomic predicate
    def __init__(self, predicate_name: str):
        self.predicate_name = predicate_name

    def visit(self, v: Visitor, i:int, keepdim) -> torch.tensor:
        # sets v.data explicitely
        # so creating a tensor for each leaf of the tree. 
                    
        if(keepdim):
            # if keepdim, return a tensor from i onwards
            v.data =tensor_log[:, i:, input.predicate_names.index(self.predicate_name)]
        else:
            # else, return only the slice for the i-th event (nan if out of bounds)
            if(i < maxlength):
                v.data = (tensor_log[:, i, input.predicate_names.index(self.predicate_name)])
            else:
                # a nan for each trace
                v.data = torch.full((1,batch_size), torch.nan)

        return v.retResult(self, i)
        
    def print(self):
        return self.predicate_name


class ComparisonTerm(Node):
    # comparison term
    def __init__(self, a, b, op):
        self.a = a
        self.b = b
        self.op = op
        self.name = op

    def visit(self, v: Visitor, i:int, keepdim) -> torch.tensor:
        
        sat_a = self.a.visit(v, i, keepdim)

        if(isinstance(self.b, float)):
            # if keepdim, use the shape of sat_a
            # else, 1 element for each case
            if(keepdim):
                sat_b = torch.full(sat_a.shape, self.b, dtype=type)
            else: 
                sat_b = torch.full((1,batch_size), self.b, dtype=type)
        else:
            sat_b = self.b.visit(v, i, keepdim)

        # setting v.data explicitely
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
       
        return v.retResult(self, i)
             
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
 
    def visit(self, v: Visitor, i:int, keepdim) -> torch.tensor:
        
        v.data = torch.sub(1, self.exp.visit(v, i, keepdim))
        return v.retResult(self, i)

    def print(self):
        return "NOT(" + self.exp.print() + ")"
    

class BoolConst(Node):
    # boolean contant
    def __init__(self, const):
        self.const = const               
        self.name = "Boolean"

    def visit(self, v: Visitor, i:int, keepdim) -> torch.tensor:
        
        # setting v.data explicitely
        # new tensor, as an leaf does
        if(keepdim):
            if(self.const==1):
                v.data = torch.ones((batch_size,maxlength), dtype=type)
            else:
                v.data = torch.zeros((batch_size,maxlength), dtype=type)
        else:
            if(self.const==1):
                v.data = torch.ones(batch_size, dtype=type)
            else:
                v.data = torch.zeros(batch_size, dtype=type)

        return v.retResult(self, i)

    def eval(self, i) -> torch.tensor:
        assert i <= maxlength, f"i exceeds maxlength ({i}>{maxlength})"

        if(self.const==1):
            return torch.ones(batch_size, dtype=type)
        else:
            return torch.zeros(batch_size, dtype=type)

    def print(self):
        return "TRUE" if self.const else "FALSE"  


class And(Node):
    # n-ary conjunction 
    def __init__(self, exprs: list):
        self.exprs = exprs
        self.name = "AND"

    def visit(self, v: Visitor, i:int, keepdim) -> torch.tensor:
        
        # evaluate first conjunct
        maxs = self.exprs[0].visit(v, i, keepdim)

        # n-ary conjunction: iterative updating of maxs (cannot use v.data) 
        for child in self.exprs[1:]:  
            maxs = torch.minimum(maxs, child.visit(v,i, keepdim))
        
        # setting v.data explicitely
        v.data = maxs
        
        return v.retResult(self, i)

    def print(self):
        strings = [ exp.print() for exp in self.exprs]
        return (" " + self.name + " ").join(strings)


class Or(Node):
    # n-ary disjunction
    def __init__(self, exprs: list):
        self.exprs = exprs
        self.name = "OR"

    def visit(self, v: Visitor, i:int, keepdim) -> torch.tensor:
        
        # evaluate first disjunct
        mins = self.exprs[0].visit(v, i, keepdim)
        
        # n-ary disjunction: iterative updating of mins (cannot use v.data)  
        for child in self.exprs[1:]:
            mins = torch.maximum(mins,child.visit(v, i, keepdim))

        # setting v.data explicitely
        v.data= mins

        return v.retResult(self, i)

    def print(self):
        strings = [ exp.print() for exp in self.exprs]
        return (" " + self.name + " ").join(strings)
    

class Implication(Node):
    # implication
    def __init__(self, a, b):
        self.a = a
        self.b = b
        # TODO: this adds an additional step. Can we just rewrite implications?
        self.exp = Or([Negate(a), b]) 
        self.name = "IMPLIES"

    def visit(self, v: Visitor, i, keepdim) -> torch.tensor:
        self.exp.visit(v, i, keepdim)
        return v.data; 

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
            # visit with i+1 to return only the next value
            v.data = self.exp.visit(v, i+1, keepdim)
        else:
            # otherwise return the tensor of evaluations, shifted left
            v.data = self.exp.visit(v, i, keepdim)
            # shift left
            v.data = torch.roll(v.data, -1) 
            # replace the last element with nan 
            v.data[:,-1] = torch.nan 
        
        # replace all nans with 0 TODO: faster if just last one?
        v.data[ v.data.isnan() ] = 0 

        return v.retResult(self, i)

    def print(self):
        return self.name + "(" + self.exp.print() + ")"
    

class WeakNext(Node):
    # weak next operator
    def __init__(self, exp):
        self.exp = exp
        self.name = "WEAK_NEXT"

    def visit(self, v: Visitor, i:int, keepdim) -> torch.tensor:

        if(not(keepdim)):
            # visit with i+1
            v.data = self.exp.visit(v, i+1, keepdim)
        else:
            v.data = self.exp.visit(v, i, keepdim)
            # shift left
            v.data = torch.roll(v.data, -1) 
            # replace the last element with nan
            v.data[:, -1] = torch.nan 

        # replace all nan with 1 TODO: faster if just last one?
        v.data[ v.data.isnan() ] = 1     

        return v.retResult(self, i)

    def print(self):
        return self.name + "(" + self.exp.print() + ")"
    

class Always(Node):
    # always operator
    def __init__(self, exp):
        self.exp = exp
        self.name = "ALWAYS"

    def visit(self, v: Visitor,i:int, keepdim) -> torch.tensor:

        # force keepdim
        self.exp.visit(v, i, True)       
        
        # shortcut
        if(not(keepdim)):
            v.data = choosemin(v.data)
        
        else:
            # update the v.data by computing the minimum for each suffix 
            # j from 0 (~instant i) to last (last=maxlength-1)
            for j in reversed(range(0, maxlength-i-1)):
                v.data[:, j] = torch.fmin(v.data[:, j], v.data[:, j + 1])
                if(debug): print(f"instant {j+i}/{maxlength-1}: {v.data[:,j]}")

        return v.retResult(self, i)
        
    def print(self):
        return self.name + "(" + self.exp.print() + ")"
    

class Eventually(Node):
    # eventually operator
    def __init__(self, exp):
        self.exp = exp
        self.name = "EVENTUALLY"

    # setting v.data explicitely
    # keepdim=True to return the evaluations of self.exp for all positions >= i
    def visit(self, v: Visitor, i:int, keepdim) -> torch.tensor:
        v.data = self.exp.visit(v, i, True)        

        # shortcut
        if(not(keepdim)):
            v.data = choosemax(v.data)

        else:
            # update the v.data by computing the maximum for each suffix 
            # j from 0 (~instant i) to last-i (last=maxlength-1)
            for j in reversed(range(0, maxlength-i-1)):
                v.data[:, j] = torch.fmax(v.data[:, j], v.data[:, j + 1])
                if(debug): print(f"instant {j+i}/{maxlength-1}: {v.data[:,j]}")

        return v.retResult(self, i)
 
    def print(self):
        return self.name + "(" + self.exp.print() + ")"


class Until(Node):
    # until operator
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.name = "UNTIL"

    def visit(self, v: Visitor,i:int,  keepdim) -> torch.tensor:
        
        # this creates new tensors for a and b
        # also setting v.data
        # forces keepdim
        sats_a = self.a.visit(v, i, True)
        sats_b = self.b.visit(v, i, True)

        # overwrites v.data TODO: we could aboid the next step (v.data=sats)
        v.data = torch.zeros(sats_a.shape, dtype=type)
        # last element as sats_b: see semantics
        v.data[:, -1] = sats_b[:, -1]

        # j from 0 (~instant i) to last (last=maxlength-1)
        for j in reversed(range(0, maxlength-i-1)):
            v.data[:, j] = torch.fmax(sats_b[:, j], torch.min(sats_a[:, j], v.data[:, j + 1]))
            if(debug): print(f"instant {j+i}/{maxlength-1}: {v.data[:,j]}")
        
        # if not(keepdim) then return the evaluation only for 0 (~instant i)
        if(not(keepdim)):
            v.data = v.data[:, 0]

        return v.retResult(self, i)

    def print(self):
        return "(" + self.a.print() + " " + self.name + " " + self.b.print() + ")"

  
class WeakUntil(Node):
    # weak until operator
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.name = "WEAK_UNTIL"

    def visit(self, v: Visitor, i:int, keepdim) -> torch.tensor:
        
        # this creates new tensors for a and b
        # also setting v.data
        # forces keepdim
        sats_a = self.a.visit(v, i, True)
        sats_b = self.b.visit(v, i, True)

        # overwrites v.data 
        v.data = torch.zeros(sats_a.shape, dtype=type)
        # last element as their maximum: see semantics
        v.data[:, -1] = torch.max(sats_a[:, -1], sats_b[:, -1])
                               
        # j from 0 (~instant i) to last (last=maxlength-1)
        for j in reversed(range(0, maxlength-i-1)):
            v.data[:, j] = torch.fmax(sats_b[:, j], torch.fmin(sats_a[:, j], v.data[:, j + 1]))
            if(debug): print(f"instant {j+i}/{maxlength-1}: {v.data[:,j]}")
        
        # if not(keepdim) then return the evaluation only for 0 (~instant i)
        if(not(keepdim)):
            v.data = v.data[:, 0]

        return v.retResult(self, i)

    def print(self):
        return "(" + self.a.print() + " " + self.name + " " + self.b.print() + ")"


class Release(Node):
    # release operator
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.name = "RELEASE"

    def visit(self, v: Visitor, i:int, keepdim) -> torch.tensor:
        
        # this creates new tensors for a and b
        # also setting v.data
        # forces keepdim
        sats_a = self.a.visit(v, i, True)
        sats_b = self.b.visit(v, i, True)

        # overwrites v.data 
        v.data = torch.zeros(sats_a.shape, dtype=type)
        # last element as sats_b: see semantics
        v.data[:, -1] = sats_b[:, -1] 
        
        # j from 0 (~instant i) to last (last=maxlength-1)
        for j in reversed(range(0, maxlength-i-1)):
            v.data[:, j] = torch.fmin(sats_b[:, j], torch.max(sats_a[:, j], v.data[:, j + 1]))
            if(debug): print(f"instant {j+i}/{maxlength-1}: {v.data[:,j]}")
        
        # if not(keepdim) then return the evaluation only for 0 (~instant i)
        if(not(keepdim)):
            v.data = v.data[:, 0]

        return v.retResult(self, i)

    def print(self):
        return "(" + self.a.print() + " " + self.name + " " + self.b.print() + ")"
    

class StrongRelease(Node):
    # strong release opeator
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.name = "STRONG_RELEASE"

    def visit(self, v: Visitor,i:int,  keepdim) -> torch.tensor:
        
        # this creates new tensors for a and b
        # also setting v.data
        # forces keepdim
        sats_a = self.a.visit(v, i, True)
        sats_b = self.b.visit(v, i, True)

        # overwrites v.data 
        v.data = torch.zeros(sats_a.shape, dtype=type)
        # last element as their minimum: see semantics
        v.data[:, -1] = torch.min(sats_a[:, -1], sats_b[:, -1])  
        
        # j from 0 (~instant i) to last (last=maxlength-1)
        for j in reversed(range(0, maxlength-i-1)):
            v.data[:, j] = torch.fmin(sats_b[:, j], torch.fmax(sats_a[:, j], v.data[:, j + 1]))
            if(debug): print(f"instant {j+i}/{maxlength-1}: {v.data[:,j]}")
            
        # if not(keepdim) then return the evaluation only for 0 (~instant i)
        if(not(keepdim)):
            v.data = v.data[:, 0]

        return v.retResult(self, i)

    def print(self):
        return "(" + self.a.print() + " " + self.name + " " + self.b.print() + ")"


class Negate:
    # negated formula. atm, no rewriting is performed
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
        elif isinstance(self.exp, ComparisonTerm):
            self.neg = ComparisonTerm(self.exp.a,self.exp.b,negComparisonOp(self.exp.op))
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
        return v.data
    
    def print(self):
        return self.neg.print()


#### Aux for class Negate ####
def negComparisonOp(comparisonOp):

    match comparisonOp:
        case "<":
            return ">="
        case ">":
            return "<="
        case "=":
            return "!="
        case "!=":
            return "="
        case "<=":
            return ">"
        case ">=":
            return "<"

#### Aux for always/eventually ####
def choosemin(tensor):
    tensor = torch.nan_to_num(tensor, nan=2) 
    tmp = torch.min(tensor, 1).values
    tmp[tmp == 2] = torch.nan
    return tmp

def choosemax(tensor):
    tensor = torch.nan_to_num(tensor, nan=-2)
    tmp = torch.max(tensor, 1).values
    tmp[tmp == -2] = torch.nan
    return tmp

