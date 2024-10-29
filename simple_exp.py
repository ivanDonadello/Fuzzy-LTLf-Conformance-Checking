import time
import csv
import numpy as np
import torch
import math
import pdb
#import matplotlib.pyplot as plt
#import pandas as pd



from FLTLf.parser import LTLfParser
from FLTLf.converter import Converter
from FLTLf import core
import input
import traceback
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-d', "--device",)

args = parser.parse_args()
core.device = args.device
core.debug = False

torch.manual_seed(0)

case_length_list = [i for i in range(500, 3500, 500)]
log_length_list = [100, 1000, 10000, 100000]
num_runs = 11

#case_length_list = [3000]
#log_length_list = [50000]


results = []
predicate_names = ["a", "b", "c"]

i = 0         # TODO togliere da input
simple_formulas = ["(X(a) >= 0.5)", "(X(a) >= 0.5)", "(WX(a) >= 0.5)", "(G(a) >= 0.5)", "(F(a) >= 0.5)", "((aUb) >= 0.5)", "((aWb) >= 0.5)", "((aRb) >= 0.5)", "((aMb) >= 0.5)"]
#simple_formulas = ["(G(a -> F(b))) > 0.2"]
#simple_formulas = ["((aM(aUb)) >= 0.1)"]

header = ["Run", "Num. Cases", "Case Length"] + simple_formulas

for log_length in log_length_list:
    for case_length in case_length_list:
        tensor_log = torch.rand(log_length, case_length, len(predicate_names))

        max_t = tensor_log.shape[1]
        batch_size = tensor_log.shape[0]

        traces = tensor_log.tolist()
        #print(traces)

        # preliminary log manipulation and padding 
        converter: Converter = Converter(predicate_names, traces)      # TODO togliere input.predicate_names, input.traces

        print("Log conversion...")
        tensor_log = converter.log2tensor(verbose=False)    # TODO togliere input.formula, non si può fare lo slicing dei predicati altrove?

        # Parsing into a formula
        parser = LTLfParser()

        for run in range(num_runs):
            results_per_formula = [run, log_length, case_length]
            for formula in simple_formulas:
                
                start = time.time()
                # prepares the log, also slicing out predicates not in the formula
                core.tensor_log, input.predicate_names = converter.slice_tensor_log(tensor_log, formula, verbose=False)
                core.tensor_log = core.tensor_log.to(core.device)

                # number of log traces
                core.batch_size = converter.batch_size

                # length of longest trace
                core.maxlength = converter.maxlength

                try:
                    pyformula = parser(formula)      # TODO togliere input.formula
                    print(f"Evaluation of {pyformula.print()} at instant {i} :")
                    visitor = core.Visitor() 
                    visitor.visit(pyformula, i)

                    #_ = pyformula.eval(i)
                except Exception as e:
                    print(traceback.format_exc())

                end = time.time()
                exec_time = end - start
                results_per_formula.append(exec_time)
                print(f"({run}, {log_length}, {case_length}) -> {exec_time}")

            results.append(results_per_formula)

with open('experiments/results/simple_conformance_check_times.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    for idx, res in enumerate(results):
        writer.writerow(res)


"""
i = 0         # TODO togliere da input
simple_formulas = ["(X(a) >= 0.5)", "(X(a) >= 0.5)", "(WX(a) >= 0.5)", "(G(a) >= 0.5)", "(F(a) >= 0.5)", "((aUb) >= 0.5)", "((aWb) >= 0.5)", "((aRb) >= 0.5)", "((aMb) >= 0.5)"]
simple_formulas = ["((aWb) >= 0.5)"]

header = ["Run", "Num. Cases", "Case Length"] + simple_formulas
  
for log_length in log_length_list:
    for case_length in case_length_list:
        tensor_log = torch.rand(log_length, case_length, len(predicate_names))

        max_t = tensor_log.shape[1]
        batch_size = tensor_log.shape[0]
        
        traces = tensor_log.tolist()
        #print(traces)

        # preliminary log manipulation and padding 
        converter: Converter = Converter(predicate_names, traces)      # TODO togliere input.predicate_names, input.traces
        
        print("Log conversion...")
        tensor_log = converter.log2tensor(verbose=False)    # TODO togliere input.formula, non si può fare lo slicing dei predicati altrove?

        # prepares the log, also slicing out predicates not in the formula
        core.tensor_log, input.predicate_names = converter.slice_tensor_log(tensor_log, formula, verbose=False)
                        
        # number of log traces
        core.batch_size = converter.batch_size
            
        # length of longest trace
        core.maxlength = converter.maxlength

        # Parsing into a formula
        parser = LTLfParser()
        
        for run in range(num_runs):
            results_per_formula = [run, log_length, case_length]
            for formula in simple_formulas:
            
                start = time.time()
                        
                try:
                    pyformula = parser(formula)      # TODO togliere input.formula
                    print(f"Evaluation of {pyformula.print()} at instant {i} :")
                    
                    visitor = core.Visitor() 
                    visitor.visit(pyformula, i)

                    _ = pyformula.eval(i)
                except Exception as e:
                    print(traceback.format_exc())
                
                end = time.time()
                exec_time = end - start
                results_per_formula.append(exec_time)
                print(f"({run}, {log_length}, {case_length}) -> {exec_time}")

            results.append(results_per_formula)
        
with open('experiments/results/simple_conformance_check_times.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    for idx, res in enumerate(results):
	    writer.writerow(res)
"""
