from lark import Transformer
from lark import Lark
from FLTLf import core
import FLTLf
import os


ParsingError = ValueError("Parsing error.")

class LTLfTransformer(Transformer):
    
    def __init__(self):
        super().__init__()

    def start(self, args):
        assert len(args) == 1
        return args[0]

    def ltlf_formula(self, args):
        assert len(args) == 1
        return args[0]

    def ltlf_implication(self, args):
        subformulas = args[::2]
        return core.Implication(subformulas[0],subformulas[1]) 

    def ltlf_comparison(self, args):
        return core.ComparisonTerm(args[0],args[2],args[1])
    
    def ltlf_or(self, args):
        subformulas = args[::2]
        return core.Or(subformulas)
        
    def ltlf_and(self, args):
        subformulas = args[::2]
        return core.And(subformulas) 

    def ltlf_until(self, args):
        return core.Until(args[0],args[2]) 
        
    def ltlf_weak_until(self, args):
        return core.WeakUntil(args[0],args[2]) 

    def ltlf_release(self, args):
        return core.Release(args[0],args[2]) 

    def ltlf_strong_release(self, args):
        return core.StrongRelease(args[0],args[2]) 

    def ltlf_always(self, args):
        return core.Always(args[1])

    def ltlf_eventually(self, args):
        return core.Eventually(args[1])

    def ltlf_next(self, args):
        return core.Next(args[1]) 

    def ltlf_weak_next(self, args):
        return core.WeakNext(args[1])
        
    def ltlf_soon(self, args):
        return core.Soon(args[1])
    
    def ltlf_lasts(self, args):
        return core.Lasts(args[4], args[2])
    
    def ltlf_b_eventually(self, args):
        return core.BoundedEventually(args[4], args[2])

    def ltlf_b_globally(self, args):
        return core.BoundedGlobally(args[4], args[2])

    def ltlf_b_within(self, args):
        return core.Within(args[4], args[2])

    def ltlf_b_until(self, args):
        return core.BoundedUntil(args[0],args[2],args[5])

    def ltlf_almost_always(self, args):
        return core.AlmostAlways(args[1])

    def ltlf_b_almost_always(self, args):
        return core.BoundedAlmostAlways(args[4],args[2])
    
    def ltlf_almost_until(self, args):
        return core.AlmostUntil(args[0],args[2]) 
    
    def ltlf_b_almost_until(self, args):
        return core.BoundedUntil(args[0],args[6],args[4])
    
    def ltlf_not(self, args):  
        return core.Negate(args[1]) 
    
    def ltlf_window(self, args):
        return int(args[0])
    
    def ltlf_real(self, args):
        return float(args[0])
    
    def ltlf_comparison_operator(self, args):
        return args[0]
        
    def ltlf_wrapped(self, args):
        if len(args) == 1:
            return args[0]
        elif len(args) == 3:
            _, formula, _ = args
            return formula
        else:
            raise ParsingError

    def ltlf_atom(self, args): 
        return args[0]

    def ltlf_predicate(self, args):
        return core.Predicate(args[0])
    
    def ltlf_true(self, args):  
        return core.BoolConst(1.0) 

    def ltlf_false(self, args):  
        return core.BoolConst(0.0) 


class LTLfParser:
    
    def __init__(self):
        
        self._transformer = LTLfTransformer()
        ltl_syntax_filepath = os.path.join(FLTLf.__path__[0], "LTLf.lark")
        self._parser = Lark(open(ltl_syntax_filepath), parser="lalr")

    def __call__(self, text):
        tree = self._parser.parse(text)
        formula = self._transformer.transform(tree)
        return formula