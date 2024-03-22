import torch
import numpy as np
import pdb

EPS = 1e-8
TH = 0.5

class TermStatic:
    def __init__(self, x):
        self.x = x

    def eval(self, t):
        return self.x

class TermDynamic:
    def __init__(self, xs):
        assert xs.dim() == 3, "Dynamic term needs 3 dims: Batch_Size x Time_Steps x Spatial_Dims"
        self.xs = xs

    def eval(self, t):
        return self.xs[:, t]
        
class Predicate:
    def __init__(self, data, predicate_name: str):
        assert data.dim() == 2, "Dynamic term needs 2 dims: Batch_Size x Time_Steps"
        self.data = data
        self.predicate_name = predicate_name

    def eval(self, t: int) -> torch.tensor:
        return self.data[:, t]

    def satisfy(self, t):
        return (self.data[:, t] > TH)
        
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

    def loss(self, t):
        return 1.0 - self.x

    def satisfy(self, t):
        return self.x > (1 - EPS)

class EQ:
    """ a == b"""
    def __init__(self, term_a, term_b):
        self.term_a = term_a
        self.term_b = term_b

    def loss(self, t):
        a = self.term_a.eval(t)
        b = self.term_b.eval(t)

        return torch.norm(a - b, dim=1)

    def satisfy(self, t):
        return (self.term_a.eval(t) == self.term_b.eval(t)).all(1)


class GEQ:
    """ a >= b """

    def __init__(self, term_a, term_b):
        self.term_a = term_a
        self.term_b = term_b
    
    def loss(self, t):
        a = self.term_a.eval(t)
        b = self.term_b.eval(t)
        return (b - a).clamp(min=0.0).sum(1)

    def satisfy(self, t):
        return (self.term_a.eval(t) >= self.term_b.eval(t)).all(1)


class LEQ:
    """ a <= b """
    def __init__(self, term_a, term_b):
        self.term_a = term_a
        self.term_b = term_b

    def loss(self, t):
        a = self.term_a.eval(t)
        b = self.term_b.eval(t)
        return (a - b).clamp(min=0.0).sum(1)

    def satisfy(self, t):
        return (self.term_a.eval(t) <= self.term_b.eval(t)).all(1)


class GT:
    """ a > b """

    def __init__(self, term_a, term_b):
        self.term_a = term_a
        self.term_b = term_b

    def loss(self, t):
        a = self.term_a.eval(t)
        b = self.term_b.eval(t)
        equality = (a == b).all(1).type(a.type()) # strict greater than, so equality penalized
        return (b - a).clamp(min=0.0).sum(1) + equality

    def satisfy(self, t):
        return (self.term_a.eval(t) > self.term_b.eval(t)).all(1)


class LT:
    """ a < b """

    def __init__(self, term_a, term_b):
        self.term_a = term_a
        self.term_b = term_b

    def loss(self, t):
        a = self.term_a.eval(t)
        b = self.term_b.eval(t)
        equality = (a == b).all(1).type(a.type()) # strict greater than, so equality penalized
        return (a - b).clamp(min=0.0).sum(1) + equality

    def satisfy(self, t):
        return (self.term_a.eval(t) < self.term_b.eval(t)).all(1)


class And:
    """ E_1 and E_2 and ... E_k"""
    def __init__(self, exprs: list):
        self.exprs = exprs

    def loss(self, t):
        losses = torch.stack([exp.loss(t) for exp in self.exprs])
        return soft_maximum(losses, 0)
        # return torch.sum(losses, dim=0, keepdim=True)
    
    def eval(self, t: int) -> torch.tensor:
        sats = torch.stack([exp.eval(t) for exp in self.exprs])
        #sats = torch.nan_to_num(sats, nan=1)
        return torch.min(sats, 0).values
        

    def satisfy(self, t):
        sats = torch.stack([exp.satisfy(t) for exp in self.exprs])
        return sats.all(0, keepdim=False)


class Or:
    """ E_1 or E_2 or .... E_k"""
    def __init__(self, exprs: list):
        self.exprs = exprs

    def loss(self, t: int):
        losses = torch.stack([exp.loss(t) for exp in self.exprs])
        return soft_minimum(losses, 0)
        # return torch.prod(losses, dim=0, keepdim=True)
    
    def eval(self, t: int) -> torch.tensor:
        sats = torch.stack([exp.eval(t) for exp in self.exprs])
        #pdb.set_trace()
        #sats = torch.nan_to_num(sats, nan=0)
        return torch.max(sats, 0).values

    def satisfy(self, t):
        sats_raw = [exp.satisfy(t) for exp in self.exprs]
        sats = torch.stack(sats_raw)
        return sats.any(0, keepdim=False)


class Implication:
    """ A -> B """
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.imp = Or([Negate(a), b])

    def loss(self, t):
        return self.imp.loss(t)
    
    def eval(self, t: int) -> torch.tensor:
        return self.imp.eval(t)

    def satisfy(self, t):
        return self.imp.satisfy(t)


class Next:
    """ N(X) """
    def __init__(self, exp, max_t: int, batch_size: int):
        self.exp = exp
        self.max_t = max_t
        self.batch_size = batch_size

    def loss(self, t):
        return self.exp.loss(t + 1)
    
    def eval(self, t: int) -> torch.tensor:
    	assert t <= self.max_t
    	if t < self.max_t - 1:
    		return self.exp.eval(t + 1)
    	else:
        	return torch.zeros(self.batch_size)

    def satisfy(self, t):
        return self.exp.satisfy(t + 1)


class WeakNext:
    """ N(X) """
    def __init__(self, exp, max_t: int, batch_size: int):
        self.exp = exp
        self.max_t = max_t
        self.batch_size = batch_size

    def loss(self, t):
        return self.exp.loss(t + 1)
    
    def eval(self, t: int) -> torch.tensor:
    	assert t <= self.max_t
    	if t < self.max_t - 1:
    		return self.exp.eval(t + 1)
    	else:
        	return torch.ones(self.batch_size)


    def satisfy(self, t):
        return self.exp.satisfy(t + 1)


class Always:
    """ Always G """
    def __init__(self, exp, max_t: int):
        self.exp = exp
        self.max_t = max_t

    def loss(self, t):
        assert t <= self.max_t
        losses = torch.stack([self.exp.loss(i) for i in range(t, self.max_t)], 1)
        # return torch.sum(losses, dim=1)
        return soft_maximum(losses, 1)

    def eval(self, t: int) -> torch.tensor:
        assert t <= self.max_t
        sats = torch.stack([self.exp.eval(i) for i in range(t, self.max_t)], 1)
        sats = torch.nan_to_num(sats, nan=1)
        return torch.min(sats, 1).values

    def satisfy(self, t):
        assert t <= self.max_t
        sats = torch.stack([self.exp.satisfy(i) for i in range(t, self.max_t)], 1)
        return sats.all(1)


class Eventually:
    """ Eventually F """
    def __init__(self, exp, max_t: int):
        self.exp = exp
        self.max_t = max_t

    def loss(self, t):
        assert t <= self.max_t
        losses = torch.stack([self.exp.loss(i) for i in range(t, self.max_t)], 1)
        return soft_minimum(losses, 1)
        
    def eval(self, t: int) -> torch.tensor:
        assert t <= self.max_t
        sats = torch.stack([self.exp.eval(i) for i in range(t, self.max_t)], 1)
        sats = torch.nan_to_num(sats, nan=0)
        return torch.max(sats, 1).values

    def satisfy(self, t):
        assert t <= self.max_t
        sats = torch.stack([self.exp.satisfy(i) for i in range(t, self.max_t)], 1)
        return sats.any(1)
    

class Release:
    def __init__(self, a, b, max_t: int):
        self.a = a
        self.b = b
        self.max_t = max_t

    def loss(self, t):
        raise NotImplementedError()
        
    def eval(self, t: int) -> torch.tensor:
	    assert t <= self.max_t
	    sats_a = torch.stack([self.a.eval(i) for i in range(t, self.max_t)], 1)
	    sats_b = torch.stack([self.b.eval(i) for i in range(t, self.max_t)], 1)
	    ys = torch.zeros(sats_a.shape)
	    ys[:, -1] = sats_b[:, -1]
	    for i in reversed(range(0, ys.shape[1] - 1)):
	    	ys[:, i] = torch.fmin(sats_b[:, i], torch.fmax(sats_a[:, i], ys[:, i + 1]))
	    return ys[:, 0]

    def satisfy(self, t):
    	# I don't think it is correct
        sats_a = torch.stack([self.a.satisfy(i) for i in range(t, self.max_t)])
        sats_b = torch.stack([self.b.satisfy(i) for i in range(t, self.max_t)])

        eventually_b = sats_b.any(dim=0, keepdim=True)
        bs_onward = torch.cumsum(sats_b.int(), dim=0)

        keep_a_until = (sats_a | bs_onward)
        return eventually_b & keep_a_until


class StrongRelease:
    def __init__(self, a, b, max_t: int):
        self.a = a
        self.b = b
        self.max_t = max_t

    def loss(self, t):
        raise NotImplementedError()
        
    def eval(self, t: int) -> torch.tensor:
	    assert t <= self.max_t
	    sats_a = torch.stack([self.a.eval(i) for i in range(t, self.max_t)], 1)
	    sats_b = torch.stack([self.b.eval(i) for i in range(t, self.max_t)], 1)
	    ys = torch.zeros(sats_a.shape)
	    ys[:, -1] = torch.fmin(sats_a[:, -1], sats_b[:, -1])
	    for i in reversed(range(0, ys.shape[1] - 1)):
	    	ys[:, i] = torch.fmin(sats_b[:, i], torch.fmax(sats_a[:, i], ys[:, i + 1]))
	    return ys[:, 0]

    def satisfy(self, t):
    	# I don't think it is correct
        sats_a = torch.stack([self.a.satisfy(i) for i in range(t, self.max_t)])
        sats_b = torch.stack([self.b.satisfy(i) for i in range(t, self.max_t)])

        eventually_b = sats_b.any(dim=0, keepdim=True)
        bs_onward = torch.cumsum(sats_b.int(), dim=0)

        keep_a_until = (sats_a | bs_onward)
        return eventually_b & keep_a_until

class WeakUntil:
    def __init__(self, a, b, max_t: int):
        self.a = a
        self.b = b
        self.max_t = max_t

    def loss(self, t):
        raise NotImplementedError()
        
    def eval(self, t: int) -> torch.tensor:
	    assert t <= self.max_t
	    sats_a = torch.stack([self.a.eval(i) for i in range(t, self.max_t)], 1)
	    sats_b = torch.stack([self.b.eval(i) for i in range(t, self.max_t)], 1)
	    ys = torch.zeros(sats_a.shape)
	    ys[:, -1] = sats_a[:, -1]
	    for i in reversed(range(0, ys.shape[1] - 1)):
	    	ys[:, i] = torch.fmax(sats_b[:, i], torch.fmin(sats_a[:, i], ys[:, i + 1]))
	    return ys[:, 0]

    def satisfy(self, t):
    	# I don't think it is correct
        sats_a = torch.stack([self.a.satisfy(i) for i in range(t, self.max_t)])
        sats_b = torch.stack([self.b.satisfy(i) for i in range(t, self.max_t)])

        eventually_b = sats_b.any(dim=0, keepdim=True)
        bs_onward = torch.cumsum(sats_b.int(), dim=0)

        keep_a_until = (sats_a | bs_onward)
        return eventually_b & keep_a_until


class Until:
    def __init__(self, a, b, max_t: int):
        self.a = a
        self.b = b
        self.max_t = max_t

    def loss(self, t):
        raise NotImplementedError()
        
    def eval(self, t: int) -> torch.tensor:
	    assert t <= self.max_t
	    sats_a = torch.stack([self.a.eval(i) for i in range(t, self.max_t)], 1)
	    sats_b = torch.stack([self.b.eval(i) for i in range(t, self.max_t)], 1)
	    ys = torch.zeros(sats_a.shape)
	    ys[:, -1] = torch.fmin(sats_a[:, -1], sats_b[:, -1])
	    for i in reversed(range(0, ys.shape[1] - 1)):
	    	ys[:, i] = torch.fmax(sats_b[:, i], torch.fmin(sats_a[:, i], ys[:, i + 1]))
	    return ys[:, 0]

    def satisfy(self, t):
    	# I don't think it is correct
        sats_a = torch.stack([self.a.satisfy(i) for i in range(t, self.max_t)])
        sats_b = torch.stack([self.b.satisfy(i) for i in range(t, self.max_t)])

        eventually_b = sats_b.any(dim=0, keepdim=True)
        bs_onward = torch.cumsum(sats_b.int(), dim=0)

        keep_a_until = (sats_a | bs_onward)
        return eventually_b & keep_a_until


class Negate:
    """ ¬X """
    def __init__(self, exp):
        self.exp = exp

        if isinstance(self.exp, LT):
            self.neg = GEQ(self.exp.term_a, self.exp.term_b)
        elif isinstance(self.exp, GT):
            self.neg = LEQ(self.exp.term_a, self.exp.term_b)
        elif isinstance(self.exp, EQ):
            self.neg = Or([LT(self.exp.term_a, self.exp.term_b), LT(self.exp.term_b, self.exp.term_a)])
        elif isinstance(self.exp, LEQ):
            self.neg = GT(self.exp.term_a, self.exp.term_b)
        elif isinstance(self.exp, GEQ):
            self.neg = LT(self.exp.term_a, self.exp.term_b)
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
        
def soft_maximum(xs, dim, p=200):
    ln_N = np.log(xs.shape[dim])
    return ((xs * p).logsumexp(dim) - ln_N) / p


def soft_minimum(xs, dim, p=200):
    ln_N = np.log(xs.shape[dim])
    return ((xs * -p).logsumexp(dim) - ln_N) / (-p)
