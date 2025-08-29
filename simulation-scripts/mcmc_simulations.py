import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bayesian_glm.model import ModelGLM


##### SIMULATION PARAMETERS 1  #####


print("\n\n[[[[[[[[[[[[[[ SIMULATION 1 ]]]]]]]]]]]]]]]]]\n\n")

n_iter = 10
kappa_values = (0.1 ,0.2 ,0.3 ,0.4 ,0.5 ,0.6 ,0.7 ,0.8 ,0.9 ,1.0)
file_name = "mcmc_order_parameters_vs_kappa_beta"

results = np.zeros((len(kappa_values)*n_iter,5))

for i, kappa in enumerate(kappa_values):
    cb_results = np.zeros(n_iter)
    print("\n***** Starting simulation for kappa = ",kappa," *****\n")
    for n in range(n_iter):
        model = ModelGLM(p=1000, n=int(1000/kappa), log_likelihood="Logistic", signal="Beta", prior="Beta", gamma=1.0)
        model.draw_sample()
        results[i*n_iter+n,0] = kappa
        results[i * n_iter + n, 1] = 1.0
        results[i * n_iter + n, 2], results[i * n_iter + n, 3], results[i * n_iter + n, 4]  = model.compute_order_parameters()
        print(" Iteration ",n," done.")
    print("\n")
    np.savetxt(file_name+".csv",results,delimiter=',')


##### SIMULATION PARAMETERS 2 #####


print("\n\n[[[[[[[[[[[[[[ SIMULATION 2 ]]]]]]]]]]]]]]]]]\n\n")

n_iter = 10
kappa_values = (0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5)
file_name = "mcmc_order_parameters_vs_kappa_bayes_opt_gaussian"

results = np.zeros((len(kappa_values)*n_iter,5))

for i, kappa in enumerate(kappa_values):
    cb_results = np.zeros(n_iter)
    print("\n***** Starting simulation for kappa = ",kappa," *****\n")
    for n in range(n_iter):
        model = ModelGLM(p=1000, n=int(1000/kappa), log_likelihood="Logistic", signal="Normal", prior="Normal", gamma=4.0, sigma=4.0)
        model.draw_sample()
        results[i*n_iter+n,0] = kappa
        results[i * n_iter + n, 1] = 1.0
        results[i * n_iter + n, 2], results[i * n_iter + n, 3], results[i * n_iter + n, 4]  = model.compute_order_parameters()
        print(" Iteration ",n," done.")
    print("\n")
    np.savetxt(file_name+".csv",results,delimiter=',')


##### SIMULATION PARAMETERS 3 #####


print("\n\n[[[[[[[[[[[[[[ SIMULATION 3 ]]]]]]]]]]]]]]]]]\n\n")

n_iter = 10
gamma_values = (0.5,0.6,0.7,0.8,0.9,1.0,1.5,2.0,2.5,3.0,5.0,7.0,9.0,11.0,13.0,15.0)
file_name = "mcmc_order_parameters_vs_snr_fixed_sigma_gaussian_rademacher"

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


##### SIMULATION PARAMETERS 4 #####


print("\n\n[[[[[[[[[[[[[[ SIMULATION 4 ]]]]]]]]]]]]]]]]]\n\n")

n_iter = 10
kappa_values = (1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0)
file_name = "mcmc_order_parameters_vs_kappa_bayes_opt_gaussian"

results = np.zeros((len(kappa_values)*n_iter,5))

for i, kappa in enumerate(kappa_values):
    cb_results = np.zeros(n_iter)
    print("\n***** Starting simulation for kappa = ",kappa," *****\n")
    for n in range(n_iter):
        model = ModelGLM(p=1000, n=int(1000/kappa), log_likelihood="Linear", signal="Normal", prior="Normal", gamma=3.0,sigma=1.0)
        model.draw_sample()
        results[i*n_iter+n,0] = kappa
        results[i * n_iter + n, 1] = 1.0
        results[i * n_iter + n, 2], results[i * n_iter + n, 3], results[i * n_iter + n, 4]  = model.compute_order_parameters()
        print(" Iteration ",n," done.")
    print("\n")
    np.savetxt(file_name+".csv",results,delimiter=',')
