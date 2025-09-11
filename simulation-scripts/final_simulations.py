import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bayesian_glm.model import ModelGLM
import mean_field_glm as mf


print("\n\n[[[[[[[[[[[[[[ SIMULATION 1 ]]]]]]]]]]]]]]]]]\n\n")

n_iter = 10
gamma_values = (0.5,0.6,0.7,0.8,0.9,1.0,1.5,2.0,2.5,3.0,5.0,7.0,9.0,11.0,13.0,15.0)
file_name = "mcmc_order_parameters_vs_snr_fixed_sigma_gaussian_rademacher"

results = np.zeros((len(gamma_values)*n_iter,5))

for i, gamma in enumerate(gamma_values):
    cb_results = np.zeros(n_iter)
    print("\n***** Starting simulation for gamma = ",gamma," *****\n")
    for n in range(n_iter):
        model = ModelGLM(p=1000, n=5000, log_likelihood="Logistic", signal="Rademacher", prior="Normal", gamma=gamma, sigma=gamma+3.0)
        model.draw_sample()
        results[i*n_iter+n,0] = 0.2
        results[i * n_iter + n, 1] = gamma
        results[i * n_iter + n, 2], results[i * n_iter + n, 3], results[i * n_iter + n, 4]  = model.compute_order_parameters()
        print(" Iteration ",n," done.")
    print("\n")
    np.savetxt(file_name+".csv",results,delimiter=',')


print("\n\n[[[[[[[[[[[[[[ SIMULATION 2 ]]]]]]]]]]]]]]]]]\n\n")

block_arguments = {"variable": "kappa",
                   "fixed_var": 0.2,
                   "num_per_var": 20,
                   "var_tuple": (0.5,0.6,0.7,0.8,0.9,1.0,1.5,2.0,2.5,3.0,5.0,7.0,9.0,11.0,13.0,15.0),
                   "signal": "Rademacher",
                   "log_likelihood": "Logistic",
                   "bayes_optimal": False,
                   "file_name": "order_parameters_vs_gamma_fixed_sigma_gaussian_rademacher"}

block = mf.block_gaussian_computation.BlockGaussianComputation(**block_arguments)
block.compute_data()


print("\n\n[[[[[[[[[[[[[[ SIMULATION 3 ]]]]]]]]]]]]]]]]]\n\n")

block_arguments = {"variable": "kappa",
                   "fixed_var": 1.0,
                   "num_per_var": 20,
                   "var_tuple": (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4),
                   "signal": "Rademacher",
                   "log_likelihood": "Logistic",
                   "file_name": "comparison_mle_kappa_gaussian_rademacher"}

block = mf.block_gaussian_computation.BlockGaussianComputation(**block_arguments)
block.compute_data()


print("\n\n[[[[[[[[[[[[[[ SIMULATION 4 ]]]]]]]]]]]]]]]]]\n\n")

block_arguments = {"variable": "snr",
                   "fixed_var": 0.1,
                   "num_per_var": 20,
                   "var_tuple": (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0),
                   "signal": "Rademacher",
                   "log_likelihood": "Logistic",
                   "file_name": "comparison_mle_gamma_variable_sigma_gaussian",
                   "bayes_optimal": True}

block = mf.block_gaussian_computation.BlockGaussianComputation(**block_arguments)
block.compute_data()


print("\n\n[[[[[[[[[[[[[[ SIMULATION 5 ]]]]]]]]]]]]]]]]]\n\n")

block_arguments = {"num_per_var": 50,
                   "var_tuple": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5),
                   "file_name": "order_parameters_vs_kappa_beta",
                   "bayes_optimal": False}

block = mf.block_beta_computation.BlockBetaComputation(**block_arguments)
block.compute_data()