import numpy as np
from matplotlib import pyplot as plt

def iterate_eqs(r1,r2,r3,gamma):
    new_cb = ((r2+1)**2*gamma**2+r3)/(r1+1)**2
    new_cbbs = ((r2+1)*gamma**2)/(r1+1)
    new_v = new_cb + 1/(r1+1)
    return new_v, new_cb, new_cbbs

def iterate_conv_eqs(v,cb,cbbs,kappa,gamma):
    new_r1 = 1/(kappa*(v-cb)+1)
    new_r2 = new_r1 -1
    new_r3 = (kappa*(gamma**2+cb-2*cbbs)+1)/(kappa*(v-cb)+1)**2
    return new_r1, new_r2, new_r3

def iterate_all_eqs(v,cb,cbbs,r1,r2,r3,kappa,gamma):
    new_v, new_cb, new_cbbs = iterate_eqs(r1,r2,r3,gamma)
    new_r1, new_r2, new_r3 = iterate_conv_eqs(v,cb,cbbs,kappa,gamma)
    return new_v, new_cb, new_cbbs, new_r1, new_r2, new_r3

def run_iterations(num_iter=10,kappa=1.0,gamma=1.0):
    v, cb, cbbs = 1.0, 1e-3, 1e-3
    r1, r2, r3 = 1.0, 1e-3, 1e-3
    for i in range(num_iter):
        v, cb, cbbs, r1, r2, r3 = iterate_all_eqs(v,cb,cbbs,r1,r2,r3,kappa,gamma)
    return v, cb, cbbs, r1, r2, r3

def create_graphs_kappa(kappa_list = [1.0],gamma=3.0):
    cb_list = []
    cbbs_list = []
    for kappa in kappa_list:
        v, cb, cbbs, r1, r2, r3 = run_iterations(10,kappa=kappa,gamma=gamma)
        cb_list.append(cb)
        cbbs_list.append(cbbs)

    # Plotting setup
    plt.style.use('ggplot')
    plt.figure(figsize=(8, 6))
    plt.plot(kappa_list, cbbs_list)
    plt.xlabel("kappa")
    plt.ylabel("c_B")
    plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.savefig("gaussian_lin_reg_cb_vs_kappa.png", dpi=1200)  # Save plot as PNG file

    mse_list = gamma**2 + np.array(cb_list) - 2 * np.array(cbbs_list)

    plt.figure(figsize=(8, 6))
    plt.plot(kappa_list, mse_list)
    plt.xlabel("kappa")
    plt.ylabel("MSE")
    plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.savefig("gaussian_lin_reg_mse_vs_kappa.png", dpi=1200)  # Save plot as PNG file

if __name__ == "__main__":
    gamma = 3.0
    kappa_list = [i/10 for i in range(1,101)]

    create_graphs_kappa(kappa_list=kappa_list,gamma=gamma)