import torch
from FLTLf.parser import LTLfParser
from FLTLf.converter import Converter
import time
import csv
import numpy as np
import math
import pdb
import matplotlib.pyplot as plt
import os


torch.manual_seed(0)
result_folder = 'results'
predicate_names = ["a", "b", "c"]
case_length_list = [i for i in range(500, 3500, 500)]
log_length_list = [100, 1000, 10000, 100000]
marker_dict = {0: "^", 1: "D", 2: "v", 3: "o"}
formulas_dict = {'complexF': "(a U b) & G(a -> X(b)) & G(b -> X(c))",
		'nextAct': "X(a)",
		'eventually': "F(a)",
		'always': "G(a)",
		'until': "a U b"}
results = ['complexF', 'nextAct', 'eventually', 'always', 'until']


for formula_name, formuala_string in formulas_dict.items():
    results = []
    for log_length in log_length_list:
        results_per_log = []
        for case_length in case_length_list:
            tensor_log = torch.rand(log_length, case_length, 3)

            max_t = tensor_log.shape[1]
            batch_size = tensor_log.shape[0]
            print(tensor_log.shape)
		    
		    # Parsing into a formula
            parser = LTLfParser(predicate_names, tensor_log, max_t, batch_size)
            formula = parser(formuala_string)

            start = time.time()
	        # evaluation
            formula.eval(0)
            #print(f"Stochastic satisfiability of the {batch_size} cases: {chain_resp.eval(0).numpy()}")
            end = time.time()
            exec_time = end - start
            results_per_log.append(exec_time)
            print(f"({log_length}, {case_length}) -> {exec_time}")

        results.append(results_per_log)
	    	
    fig, ax = plt.subplots()
    ax.set(xlabel='# Events', ylabel='Time (log. secs)', title='Conformance checking performance')
    ax.grid()
 
    with open(os.path.join(result_folder, f'conformance_check_times_{formula_name}.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([""] + case_length_list)
        for idx, res in enumerate(results):
            writer.writerow([log_length_list[idx]] + res)
            ax.plot(case_length_list, np.log10(res), label=f"{log_length_list[idx]} traces", marker=marker_dict[idx])            

    plt.legend(loc='upper left')
    plt.tight_layout()
    fig.savefig(os.path.join(result_folder, f'conformance_check_times_{formula_name}.pdf'))
    plt.show()

print(results)




