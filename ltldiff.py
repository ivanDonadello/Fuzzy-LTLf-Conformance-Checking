import torch
import numpy as np
import pdb

        
class Predicate:
    def __init__(self, data, predicate_name: str):
        assert data.dim() == 2, "Dynamic term needs 2 dims: Batch_Size x Time_Steps"
        self.data = data
        self.predicate_name = predicate_name

    def eval(self, t: int) -> torch.tensor:
        return self.data[:, t]

        
class NegPredicate:
    def __init__(self, data, predicate_name):
        assert data.dim() == 2, "Dynamic term needs 2 dims: Batch_Size x Time_Steps"
        self.data = data
        self.predicate_name = predicate_name

    def eval(self, t):
        return 1 - self.data[:, t]


class BoolConst:
    def __init__(self, x):
        self.x = x.float() ## True is 1, False is 0

    def satisfy(self, t):
        return self.x > (1 - EPS)


class And:
    """ E_1 and E_2 and ... E_k"""
    def __init__(self, exprs: list):
        self.exprs = exprs
    
    def eval(self, t: int) -> torch.tensor:
        sats = torch.stack([exp.eval(t) for exp in self.exprs])
        #sats = torch.nan_to_num(sats, nan=1)
        return torch.min(sats, 0).values
        

class Or:
    """ E_1 or E_2 or .... E_k"""
    def __init__(self, exprs: list):
        self.exprs = exprs
    
    def eval(self, t: int) -> torch.tensor:
        sats = torch.stack([exp.eval(t) for exp in self.exprs])
        #pdb.set_trace()
        #sats = torch.nan_to_num(sats, nan=0)
        return torch.max(sats, 0).values


class Implication:
    """ A -> B """
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.imp = Or([Negate(a), b])
    
    def eval(self, t: int) -> torch.tensor:
        return self.imp.eval(t)


class Next:
    """ N(X) """
    def __init__(self, exp, max_t: int, batch_size: int):
        self.exp = exp
        self.max_t = max_t
        self.batch_size = batch_size
    
    def eval(self, t: int) -> torch.tensor:
    	assert t <= self.max_t
    	if t < self.max_t - 1:
    		return self.exp.eval(t + 1)
    	else:
        	return torch.zeros(self.batch_size)


class WeakNext:
    """ N(X) """
    def __init__(self, exp, max_t: int, batch_size: int):
        self.exp = exp
        self.max_t = max_t
        self.batch_size = batch_size
    
    def eval(self, t: int) -> torch.tensor:
    	assert t <= self.max_t
    	if t < self.max_t - 1:
    		return self.exp.eval(t + 1)
    	else:
        	return torch.ones(self.batch_size)


class Always:
    """ Always G """
    def __init__(self, exp, max_t: int):
        self.exp = exp
        self.max_t = max_t

    def eval(self, t: int) -> torch.tensor:
        assert t <= self.max_t
        sats = torch.stack([self.exp.eval(i) for i in range(t, self.max_t)], 1)
        sats = torch.nan_to_num(sats, nan=1)
        return torch.min(sats, 1).values


class Eventually:
    """ Eventually F """
    def __init__(self, exp, max_t: int):
        self.exp = exp
        self.max_t = max_t
        
    def eval(self, t: int) -> torch.tensor:
        assert t <= self.max_t
        sats = torch.stack([self.exp.eval(i) for i in range(t, self.max_t)], 1)
        sats = torch.nan_to_num(sats, nan=0)
        return torch.max(sats, 1).values


class Release:
    def __init__(self, a, b, max_t: int):
        self.a = a
        self.b = b
        self.max_t = max_t
        
    def eval(self, t: int) -> torch.tensor:
	    assert t <= self.max_t
	    sats_a = torch.stack([self.a.eval(i) for i in range(t, self.max_t)], 1)
	    sats_b = torch.stack([self.b.eval(i) for i in range(t, self.max_t)], 1)
	    ys = torch.zeros(sats_a.shape)
	    ys[:, -1] = sats_b[:, -1]
	    for i in reversed(range(0, ys.shape[1] - 1)):
	    	ys[:, i] = torch.fmin(sats_b[:, i], torch.fmax(sats_a[:, i], ys[:, i + 1]))
	    return ys[:, 0]


class StrongRelease:
    def __init__(self, a, b, max_t: int):
        self.a = a
        self.b = b
        self.max_t = max_t
        
    def eval(self, t: int) -> torch.tensor:
	    assert t <= self.max_t
	    sats_a = torch.stack([self.a.eval(i) for i in range(t, self.max_t)], 1)
	    sats_b = torch.stack([self.b.eval(i) for i in range(t, self.max_t)], 1)
	    ys = torch.zeros(sats_a.shape)
	    ys[:, -1] = torch.fmin(sats_a[:, -1], sats_b[:, -1])
	    for i in reversed(range(0, ys.shape[1] - 1)):
	    	ys[:, i] = torch.fmin(sats_b[:, i], torch.fmax(sats_a[:, i], ys[:, i + 1]))
	    return ys[:, 0]


class WeakUntil:
    def __init__(self, a, b, max_t: int):
        self.a = a
        self.b = b
        self.max_t = max_t
        
    def eval(self, t: int) -> torch.tensor:
	    assert t <= self.max_t
	    sats_a = torch.stack([self.a.eval(i) for i in range(t, self.max_t)], 1)
	    sats_b = torch.stack([self.b.eval(i) for i in range(t, self.max_t)], 1)
	    ys = torch.zeros(sats_a.shape)
	    ys[:, -1] = sats_a[:, -1]
	    for i in reversed(range(0, ys.shape[1] - 1)):
	    	ys[:, i] = torch.fmax(sats_b[:, i], torch.fmin(sats_a[:, i], ys[:, i + 1]))
	    return ys[:, 0]


class Until:
    def __init__(self, a, b, max_t: int):
        self.a = a
        self.b = b
        self.max_t = max_t
        
    def eval(self, t: int) -> torch.tensor:
	    assert t <= self.max_t
	    sats_a = torch.stack([self.a.eval(i) for i in range(t, self.max_t)], 1)
	    sats_b = torch.stack([self.b.eval(i) for i in range(t, self.max_t)], 1)
	    ys = torch.zeros(sats_a.shape)
	    ys[:, -1] = torch.fmin(sats_a[:, -1], sats_b[:, -1])
	    for i in reversed(range(0, ys.shape[1] - 1)):
	    	ys[:, i] = torch.fmax(sats_b[:, i], torch.fmin(sats_a[:, i], ys[:, i + 1]))
	    return ys[:, 0]


class Negate:
    """ ¬X """
    def __init__(self, exp):
        self.exp = exp

        if isinstance(self.exp, And):
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
        elif isinstance(self.exp, Eventually):
            self.neg = Always(Negate(self.exp.exp), self.exp.max_t)
        elif isinstance(self.exp, Always):
            self.neg = Eventually(Negate(self.exp.exp), self.exp.max_t)
        elif isinstance(self.exp, Predicate):
            self.neg = NegPredicate(self.exp.data, f"neg_{self.exp.predicate_name}")
        else:
            assert False, 'Class not supported %s' % str(type(exp))

    def loss(self, t):
        return self.neg.loss(t)
        
    def eval(self, t: int) -> torch.tensor:
        return self.neg.eval(t)

    def satisfy(self, t):
        return self.neg.satisfy(t)
