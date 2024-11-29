import time
import torch
import os
import pandas as pd
import input
import traceback
import argparse


from FLTLf.parser import LTLfParser
from FLTLf.converter import Converter
from FLTLf import core

parser = argparse.ArgumentParser()

parser.add_argument('-d', "--device", default="cpu")

args = parser.parse_args()
core.device = args.device
core.debug = False

torch.manual_seed(0)

num_runs = 10
results = {"Run" : [], "Num. Cases" : [], "Case Length" : []}
batch_sizes_column = []
maxlengths_column = []
run_column = []

i = 0 
maxlengths = [3000]
batch_sizes = [100, 1000, 10000, 100000]
models_lengths = [25, 50, 75, 100]
num_AP = 50
predicate_names = [f"p{i}" for i in range(num_AP)]
number_of_formulas = 100

simple_declare_formulas = []
declare_models = []

# loading all DECLARE formulas
with open(os.path.join("data", f"{num_AP}_AP_{number_of_formulas}_constraints.csv"), 'r') as file:
    for line in file:
        formula = line.strip().split(",")[1]
        if formula != 'formula':
            simple_declare_formulas.append(formula)

# Aggregation with ANDs
for num_constraints in models_lengths:
    declare_model = "(" + ") & (".join(simple_declare_formulas[:num_constraints]) + ")"
    declare_models.append(declare_model)


results_per_formula = {formula : [] for formula in declare_models}

#verbose printouts
verbose = False
#save results on a csv
savefile = True
#padding needed?
skippadding = True

#for every combination of maxlength,batchsize
for batch_size in batch_sizes:
    for maxlength in maxlengths:
        
        run_column.extend([i for i in range(num_runs)])
        batch_sizes_column.extend([batch_size]*num_runs)
        maxlengths_column.extend([maxlength]*num_runs)

        print("------times-------")
        start = time.time()

        #generate a random tensor log
        tensor_log = torch.rand(batch_size, maxlength, len(predicate_names), dtype=torch.half)
        rand = time.time()
        print(f"RANDOM GEN time {rand - start}")

        #padding
        converter = Converter(predicate_names)    
        tensor_log = converter.log2tensor(tensor_log, verbose, skippadding) 

        if not(skippadding):
            print(f"PADDING time {time.time() - rand}")
        else:
            print(f"PADDING skipped")        

        # Parsing into a formula
        parser = LTLfParser()
        
        #for each formula
        
        for formula in declare_models:
      
            start = time.time()
                
            #slicing of predicates not in formula
            core.tensor_log, input.predicate_names = converter.slice_tensor_log(tensor_log, formula, verbose)
            core.tensor_log = core.tensor_log.to(core.device)
                
            #setting dimensions for practical access
            core.batch_size = converter.batch_size
            core.maxlength = converter.maxlength

            ready = time.time()
            print(f"SLICING TIME {ready - start}")

            try:
                pyformula = parser(formula)     
            except Exception as e:
                print(traceback.format_exc())

            print(f"PARSING TIME {time.time() - ready}")
                            
                
            for run in range(num_runs):

                start = time.time()

                visitor = core.Visitor()

                print("------------------")
                print(f"Evaluation of {pyformula.print()} at instant {i} :")
                try:
                    visitor.visit(pyformula, i)

                except Exception as e:
                    print(traceback.format_exc())

                end = time.time()
                exec_time = end - start

                if savefile:
                    results_per_formula[formula].append(exec_time)

                print(f"({run}, {core.batch_size}, {core.maxlength}) -> {exec_time}")


results["Run"] = run_column
results["Num. Cases"] = batch_sizes_column
results["Case Length"] = maxlengths_column

results_df = pd.DataFrame.from_dict(results_per_formula)
settings_df = pd.DataFrame.from_dict(results)
results_df = pd.concat([settings_df, results_df], axis=1)
results_df.rename(columns={declare_models[0] : "25 constraints", declare_models[1] : "50 constraints", declare_models[2] : "75 constraints", declare_models[3]: "100 constraints"}, inplace=True)

if savefile:
    results_df.to_csv(os.path.join('results', f'DECLARE_conformance_check_times_{core.device.upper()}.csv'))
