import numpy as np
import matplotlib.pyplot as plt

file_path = "./simulation-results/logistic_beta_beta_kappa_graph_data.csv"
data = np.loadtxt(file_path, delimiter=',')

kappa = 1 / data[:, 0]  # x-values
c_b = data[:, 1]        # y-values
y_error = data[:, 2]   # y-error

plt.style.use('ggplot')
plt.figure(figsize=(8, 6))

plt.errorbar(kappa, c_b, yerr=y_error, fmt='o', color='royalblue', capsize=7)

plt.xlabel("kappa", fontsize=14)
plt.ylabel("c_B", fontsize=14)
plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
plt.minorticks_on()

plt.savefig("logistics_beta_beta_order_parameters.png", dpi=600)
plt.show()