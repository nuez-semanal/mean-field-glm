import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mean_field_glm.mean_field_MLE import LogisticMeanFieldMLE

kappa_list = [0.1,0.15,0.2,0.25,0.3,0.35,0.4]

solutions_stats = np.zeros([len(kappa_list),4])

for i, kappa in enumerate(kappa_list):
    iterations = np.zeros([5,3])
    for j in range(5):
        model = LogisticMeanFieldMLE(kappa=kappa,gamma=1.0)
        iterations[j,:] = model.find_solutions()
    solutions_stats[i,0] = np.mean(iterations[:,0])
    solutions_stats[i,1] = np.std(iterations[:,0])
    solutions_stats[i,2] = np.mean(iterations[:,1])
    solutions_stats[i,3] = np.std(iterations[:,1])
    print(solutions_stats,"\n")
    print("\n[ ----- Done with kappa = ",kappa," ----- ]\n")

np.savetxt("MLE_simulations_kappa.csv",solutions_stats,delimiter=",")