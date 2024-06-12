#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implementation of the FLTLf parser."""

from lark import Lark, Transformer
#import ltlf2dfa
#from ltlf2dfa.helpers import ParsingError
#from ltlf2dfa.parser.pl import PLTransformer
import FLTLf
from FLTLf import core as fltlf
import os


ParsingError = ValueError("Parsing error.")

class LTLfTransformer(Transformer):
    """LTLf Transformer."""

    def __init__(self,predicates, tensor_log, max_t, batch_size):
        """Initialize."""
        super().__init__()
        self.predicates = predicates
        self.tensor_log = tensor_log
        self.max_t = max_t
        self.batch_size = batch_size

    def start(self, args):
        """Entry point."""
        assert len(args) == 1
        return args[0]

    def ltlf_formula(self, args):
        """Parse FLTLf formula."""
        assert len(args) == 1
        return args[0]

    def ltlf_implication(self, args):
        """Parse FLTLf Implication."""
        
        if len(args) == 1:
            return args[0] 
        elif (len(args) - 1) % 2 == 0:  
            subformulas = args[::2]
            return fltlf.Implication(subformulas[0],subformulas[1]) 
        else:
            raise ParsingError

    def ltlf_or(self, args):
        """Parse FLTLf Or."""
        if len(args) == 1: 
            return args[0]
        elif (len(args) - 1) % 2 == 0:
            subformulas = args[::2]
            return fltlf.Or(subformulas)
        else:
            raise ParsingError

    def ltlf_and(self, args):
        """Parse FLTLf And."""
        if len(args) == 1:
            return args[0]
        elif (len(args) - 1) % 2 == 0:
            subformulas = args[::2]
            return fltlf.And(subformulas) 
        else:
            raise ParsingError

    def ltlf_until(self, args):
        """Parse FLTLf Until."""
        if len(args) == 1:
            return args[0]
        elif (len(args) - 1) % 2 == 0:
            subformulas = args[::2] 
            return fltlf.Until(subformulas[0],subformulas[1], self.max_t) 
        else:
            raise ParsingError
        
    def ltlf_weak_until(self, args):
        """Parse FLTLf Weak Until."""
        if len(args) == 1:
            return args[0]
        elif (len(args) - 1) % 2 == 0:
            subformulas = args[::2]
            return fltlf.WeakUntil(subformulas[0],subformulas[1], self.max_t) 
        else:
            raise ParsingError

    def ltlf_release(self, args):
        """Parse FLTLf Release."""
        if len(args) == 1:
            return args[0]
        elif (len(args) - 1) % 2 == 0:
            subformulas = args[::2]
            return fltlf.Release(subformulas[0],subformulas[1], self.max_t) 
        else:
            raise ParsingError

    def ltlf_strong_release(self, args):
        """Parse FLTLf StrongRelease."""
        if len(args) == 1:
            return args[0]
        elif (len(args) - 1) % 2 == 0:
            subformulas = args[::2]
            return fltlf.StrongRelease(subformulas[0],subformulas[1], self.max_t) 
        else:
            raise ParsingError

    def ltlf_always(self, args):
        """Parse FLTLf Always."""       
        if len(args) == 1:
            return args[0] 
        else:
            f = args[-1] 
            for _ in args[:-1]:
                f = fltlf.Always(f, self.max_t)
            return f

    def ltlf_eventually(self, args):
        """Parse FLTLf Eventually."""
        if len(args) == 1:
            return args[0]
        else:
            f = args[-1]
            for _ in args[:-1]:
                f = fltlf.Eventually(f, self.max_t) 
            return f

    def ltlf_next(self, args):
        """Parse FLTLf Next."""
        if len(args) == 1:
            return args[0]
        else:
            f = args[-1]
            for _ in args[:-1]:
                f = fltlf.Next(f, self.max_t, self.batch_size) 
            return f

    def ltlf_weak_next(self, args):
        """Parse FLTLf Weak Next."""
        if len(args) == 1:
            return args[0]
        else:
            f = args[-1]
            for _ in args[:-1]:
                f = fltlf.WeakNext(f, self.max_t, self.batch_size)
            return f

    def ltlf_not(self, args):  
        """Parse FLTLf Not."""
        f = args[-1]
        for _ in args[:-1]:
            f = fltlf.Negate(f) 
        return f
        

    def ltlf_wrapped(self, args):
        """Parse FLTLf wrapped formula."""
        if len(args) == 1:
            return args[0]
        elif len(args) == 3:
            _, formula, _ = args
            return formula
        else:
            raise ParsingError

    def ltlf_atom(self, args): 
        """Parse FLTLf Atom."""
        assert len(args) == 1
        return args[0]
    
    def ltlf_true(self, args):  
        """Parse FLTLf True."""
        return fltlf.BoolConst(1, self.batch_size) 

    def ltlf_false(self, args):  
        """Parse FLTLf False."""
        return fltlf.BoolConst(0, self.batch_size) 

    def ltlf_symbol(self, args):
        """Parse FLTLf Symbol."""
        assert len(args) == 1
        symbol = str(args[0])
        return fltlf.Predicate(self.tensor_log[:, :, self.predicates.index(symbol)], symbol)

class LTLfParser:
    """FLTLf Parser class."""

    def __init__(self,predicates,tensor_log,max_t,batch_size_var):
        """Initialize."""
        self._transformer = LTLfTransformer(predicates, tensor_log, max_t, batch_size_var)
        ltl_syntax_filepath = os.path.join(FLTLf.__path__[0], "LTLf.lark")
        self._parser = Lark(open(ltl_syntax_filepath), parser="lalr")

    def __call__(self, text):
        """Call."""
        tree = self._parser.parse(text)
        formula = self._transformer.transform(tree)
        return formula
