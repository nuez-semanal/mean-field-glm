import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

gamma_list = [0.1,0.5,1.0,3.0,5.0,7.0,9.0]

parameters_stats = np.zeros([len(gamma_list),4])

simulations = np.loadtxt("./simulation-results/comparison_mle_snr_variable_sigma_gaussian.csv",delimiter=",")

for i, gamma in enumerate(gamma_list):
    iterations = simulations[i*5:(i+1)*5,:]
    parameters_stats[i,0] = np.mean(iterations[:,7]/(iterations[:,6]+1/(gamma+3.0)**2))
    parameters_stats[i,1] = np.std(iterations[:,7]/(iterations[:,6]+1/(gamma+3.0)**2))
    parameters_stats[i,2] = np.mean(iterations[:,8]/(iterations[:,6]+1/(gamma+3.0)**2))
    parameters_stats[i,3] = np.std(iterations[:,8]/(iterations[:,6]+1/(gamma+3.0)**2))
    print("\n[ ----- Done with gamma = ",gamma," ----- ]\n")

np.savetxt("Bayes_simulations_gamma.csv",parameters_stats,delimiter=",")