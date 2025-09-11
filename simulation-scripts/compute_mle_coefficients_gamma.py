import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mean_field_glm.mean_field_MLE import LogisticMeanFieldMLE

gamma_list = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]

solutions_stats = np.zeros([len(gamma_list),4])

for i, gamma in enumerate(gamma_list):
    iterations = np.zeros([5,3])
    for j in range(5):
        model = LogisticMeanFieldMLE(kappa=0.1,gamma=gamma)
        iterations[j,:] = model.find_solutions()
    solutions_stats[i,0] = np.mean(iterations[:,0])
    solutions_stats[i,1] = np.std(iterations[:,0])
    solutions_stats[i,2] = np.mean(iterations[:,1])
    solutions_stats[i,3] = np.std(iterations[:,1])
    print("\n[ ----- Done with gamma = ",gamma," ----- ]\n")

np.savetxt("MLE_simulations_gamma.csv",solutions_stats,delimiter=",")