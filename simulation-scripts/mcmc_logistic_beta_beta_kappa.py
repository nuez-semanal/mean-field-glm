import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bayesian_glm.model import ModelGLM

commit_message = "Automated simulation commit"

def commit_results():
    os.system("git add .")
    os.system(f'git commit -m "{commit_message}"')
    os.system("git push")


##### SIMULATION PARAMETERS 1 #####

n_iter = 10
kappa_values = (0.1 ,0.2 ,0.3 ,0.4 ,0.5 ,0.6 ,0.7 ,0.8 ,0.9 ,1.0)
file_name = "mcmc_order_parameters_vs_kappa_beta"

results = np.zeros((len(kappa_values)*n_iter,5))

for i, kappa in enumerate(kappa_values):
    cb_results = np.zeros(n_iter)
    print("\n***** Starting simulation for kappa = ",kappa," *****\n")
    for n in range(n_iter):
        model = ModelGLM(p=1000, n=int(1000/kappa), log_likelihood="Logistic", signal="Beta", prior="Beta", snr=1.0)
        model.draw_sample()
        results[i*n_iter+n,0] = kappa
        results[i * n_iter + n, 1] = 1.0
        results[i * n_iter + n, 2], results[i * n_iter + n, 3], results[i * n_iter + n, 4]  = model.compute_order_parameters()
        print(" Iteration ",n," done.")
    print("\n")
    np.savetxt(file_name+".csv",results,delimiter=',')


##### SIMULATION PARAMETERS 2 #####

n_iter = 10
kappa_values = (0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5)
file_name = "mcmc_order_parameters_vs_kappa_bayes_opt_gaussian"

results = np.zeros((len(kappa_values)*n_iter,5))

for i, kappa in enumerate(kappa_values):
    cb_results = np.zeros(n_iter)
    print("\n***** Starting simulation for kappa = ",kappa," *****\n")
    for n in range(n_iter):
        model = ModelGLM(p=1000, n=int(1000/kappa), log_likelihood="Logistic", signal="Normal", prior="Normal", snr=4.0)
        model.draw_sample()
        results[i*n_iter+n,0] = kappa
        results[i * n_iter + n, 1] = 1.0
        results[i * n_iter + n, 2], results[i * n_iter + n, 3], results[i * n_iter + n, 4]  = model.compute_order_parameters()
        print(" Iteration ",n," done.")
    print("\n")
    np.savetxt(file_name+".csv",results,delimiter=',')


##### SIMULATION PARAMETERS 3 #####

n_iter = 10
kappa_values = (0.5,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,1.0)
file_name = "mcmc_order_parameters_vs_kappa_bayes_opt_gaussian"

results = np.zeros((len(kappa_values)*n_iter,5))

for i, kappa in enumerate(kappa_values):
    cb_results = np.zeros(n_iter)
    print("\n***** Starting simulation for kappa = ",kappa," *****\n")
    for n in range(n_iter):
        model = ModelGLM(p=1000, n=int(1000/kappa), log_likelihood="Logistic", signal="Normal", prior="Normal", snr=4.0)
        model.draw_sample()
        results[i*n_iter+n,0] = kappa
        results[i * n_iter + n, 1] = 1.0
        results[i * n_iter + n, 2], results[i * n_iter + n, 3], results[i * n_iter + n, 4]  = model.compute_order_parameters()
        print(" Iteration ",n," done.")
    print("\n")
    np.savetxt(file_name+".csv",results,delimiter=',')


##### SIMULATION PARAMETERS 4 #####

n_iter = 10
kappa_values = (0.5,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,1.0)
file_name = "mcmc_order_parameters_vs_kappa_bayes_opt_gaussian"

results = np.zeros((len(kappa_values)*n_iter,5))

for i, kappa in enumerate(kappa_values):
    cb_results = np.zeros(n_iter)
    print("\n***** Starting simulation for kappa = ",kappa," *****\n")
    for n in range(n_iter):
        model = ModelGLM(p=1000, n=int(1000/kappa), log_likelihood="Logistic", signal="Normal", prior="Normal", snr=4.0)
        model.draw_sample()
        results[i*n_iter+n,0] = kappa
        results[i * n_iter + n, 1] = 1.0
        results[i * n_iter + n, 2], results[i * n_iter + n, 3], results[i * n_iter + n, 4]  = model.compute_order_parameters()
        print(" Iteration ",n," done.")
    print("\n")
    np.savetxt(file_name+".csv",results,delimiter=',')
