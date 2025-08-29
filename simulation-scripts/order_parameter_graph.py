import numpy as np
import matplotlib.pyplot as plt

file_path = "./simulation-results/order_parameters_vs_kappa_beta_old.csv"
data = np.loadtxt(file_path, delimiter=',')

kappa = 1 / data[:, 0]
c_b = data[:, 1]
y_error = data[:, 2]

plt.style.use('ggplot')
plt.figure(figsize=(8, 6))

plt.errorbar(kappa, c_b, yerr=y_error, fmt='o', color='royalblue', capsize=7)

plt.xlabel("kappa", fontsize=14)
plt.ylabel("c_B", fontsize=14)
plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
plt.minorticks_on()

plt.savefig("order_parameters_vs_kappa_beta_C_B.png", dpi=1200)
plt.show()