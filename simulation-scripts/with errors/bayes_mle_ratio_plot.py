import matplotlib.pyplot as plt
import numpy as np

# Simulation results
data = np.array([
    [0.1, 0.731337980500159, 0.61425270559267],
    [0.5, 0.746156360175268, 0.620610859776302],
    [1, 0.762914302875847, 0.629160866207447],
    [5, 0.857099098042936, 0.693705808908152],
    [10, 0.924195421384531, 0.745699859043615],
    [20, 0.990774661501211, 0.804207537570587]
])

snr = data[:, 0]
ratio_alpha = data[:, 1]
ratio_sigma = data[:, 2]

plt.figure(figsize=(8, 5))
plt.plot(snr, ratio_alpha, marker='o', label='ratio alpha', linewidth=2)
plt.plot(snr, ratio_sigma, marker='s', label='ratio sigma', linewidth=2)

plt.xscale('log')
plt.xlabel('SNR', fontsize=12)
plt.ylabel('ratio', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(fontsize=12)
plt.tight_layout()

plt.savefig('ratio_vs_snr_log_scale.png', dpi=600)

plt.show()