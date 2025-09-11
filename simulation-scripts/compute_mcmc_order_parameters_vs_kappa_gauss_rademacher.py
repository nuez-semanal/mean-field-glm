import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bayesian_glm.model import ModelGLM

n_per_kappa = 5

kappa_list = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

results_list = np.ones((len(kappa_list)*n_per_kappa,5))

for i, k in enumerate(kappa_list):
    for j in range(n_per_kappa):
        model = ModelGLM(
            p=500,
            n=int(500/k),
            log_likelihood="Logistic",
            signal="Rademacher",
            prior="Normal",
            gamma=1.0,
            sigma=1.0
        )
        model.draw_sample()
        results = np.array(model.compute_order_parameters())
        results_list[i*n_per_kappa+j,0] = k
        results_list[i*n_per_kappa+j,2:] = results
        print("\nSimulation (",i,j,") ready!\n")

np.savetxt("mcmc_order_parameters_vs_kappa_gauss_rademacher.csv",results_list,delimiter=",")