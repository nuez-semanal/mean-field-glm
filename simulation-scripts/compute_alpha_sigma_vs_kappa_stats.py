import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

kappa_list = [0.1,0.15,0.2,0.25,0.3,0.35,0.4]

parameters_stats = np.zeros([len(kappa_list),4])

simulations = np.loadtxt("./simulation-results/comparison_mle_kappa_gaussian_rademacher.csv",delimiter=",")

gamma = 1.0

for i, kappa in enumerate(kappa_list):
    iterations = simulations[i*5:(i+1)*5,:]
    parameters_stats[i,0] = np.mean(iterations[:,7]/(iterations[:,6]+1/(gamma+3.0)**2))
    parameters_stats[i,1] = np.std(iterations[:,7]/(iterations[:,6]+1/(gamma+3.0)**2))
    parameters_stats[i,2] = np.mean(iterations[:,8]/(iterations[:,6]+1/(gamma+3.0)**2))
    parameters_stats[i,3] = np.std(iterations[:,8]/(iterations[:,6]+1/(gamma+3.0)**2))
    print("\n[ ----- Done with kappa = ",kappa," ----- ]\n")

np.savetxt("Bayes_simulations_kappa.csv",parameters_stats,delimiter=",")