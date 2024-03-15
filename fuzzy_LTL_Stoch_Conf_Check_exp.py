import ltldiff as ltd
import time
import csv
import numpy as np
import torch
import math
import pdb
import matplotlib.pyplot as plt


predicate_names = ["excavate", "concrete", "scaffolding"]
case_length_list = [i for i in range(500, 3500, 500)]
log_length_list = [100, 1000, 10000, 100000]
marker_dict = {0: "^", 1: "D", 2: "v", 3: "o"}
results = []

for log_length in log_length_list:
	results_per_log = []
	for case_length in case_length_list:
		tensor_log = torch.rand(log_length, case_length, 3)
	
		max_t = tensor_log.shape[1]
		batch_size = tensor_log.shape[0]
		print(tensor_log.shape)

		pred_excavate = ltd.Predicate(tensor_log[:, :, 0], predicate_name=predicate_names[0])
		pred_concrete = ltd.Predicate(tensor_log[:, :, 1], predicate_name=predicate_names[1])
		pred_scaffolding = ltd.Predicate(tensor_log[:, :, 2], predicate_name=predicate_names[2])

		start = time.time()
		# definition crisp LTLf formula G(excavate -> X(concrete)) AND G(concrete -> X(scaffolding))
		complexF = ltd.And([ltd.Until(pred_excavate, pred_concrete, max_t), ltd.Always(ltd.Implication(pred_excavate, ltd.Next(pred_concrete, max_t, batch_size)), max_t),
			ltd.Always(ltd.Implication(pred_concrete, ltd.Next(pred_scaffolding, max_t, batch_size)), max_t)])
			
		nextOp = ltd.Next(pred_excavate, max_t, batch_size)
		always = ltd.Always(pred_excavate, max_t)
		eventually = ltd.Eventually(pred_excavate, max_t)
		#until = ltd.Until(pred_excavate, pred_concrete, max_t)
			
		# evaluation
		eventually.eval(0)
		#print(f"Stochastic satisfiability of the {batch_size} cases: {chain_resp.eval(0).numpy()}")
		end = time.time()
		exec_time = end - start
		results_per_log.append(exec_time)
		print(f"({log_length}, {case_length}) -> {exec_time}")
	results.append(results_per_log)
	
print(results)

fig, ax = plt.subplots()
ax.set(xlabel='# Events', ylabel='Time (secs)', title='Conformance checking (complex formula)')
ax.grid()

with open('conformance_check_times_F.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([""] + case_length_list)
    for idx, res in enumerate(results):
	    writer.writerow([log_length_list[idx]] + res)
	    ax.plot(case_length_list, np.log10(res), label=f"{log_length_list[idx]} traces", marker=marker_dict[idx])

plt.legend(loc='upper left')
plt.tight_layout()
#fig.savefig("conformance_check_times_U_2Chain_resp.pdf")
fig.savefig("conformance_check_times_F.pdf")
plt.show()