import numpy as np
from matplotlib import pyplot as plt
from block_computation import BlockComputation

class ComputeMseGraph(BlockComputation):
    def __init__(self,kappa_list = [1.0], num_per_kappa = 5, delta = 1.0, snr = 1.0,
                 prior = "Normal", signal = "Normal"):
        super().__init__(kappa_list, num_per_kappa, delta, snr, prior, signal)

    def plot_graph_mse(self,save=False):
        self.compute_data()
        self.save_data()

        n_kappa = len(self.kappa_list)

        stats = np.zeros((n_kappa,5))

        for i in range(n_kappa):
            block_data = self.data[i*self.num_per_kappa:(i+1)*self.num_per_kappa,:]
            stats[i,0] = 1/block_data[0,0]
            stats[i,1] = np.mean(block_data[:,2])
            stats[i,2] = np.std(block_data[:,2])
            stats[i,3] = np.mean(block_data[:,3])
            stats[i,4] = np.std(block_data[:,3])

        x = stats[:,0]
        y = 1 + stats[:,1] - 2 * stats[:,3]

        x_error = np.zeros_like(x)
        y_error = stats[:,2] + 2 * stats[:,4]

        if save:
            graph_data = np.array([x, y, y_error]).transpose()
            np.savetxt("MSE_graph_data.csv", graph_data, delimiter=",")

        plt.errorbar(x, y, xerr=x_error, yerr=y_error, fmt='o', elinewidth=3, capsize=0, alpha=0.6)

        plt.title('Mean square error as a function of kappa')
        plt.xlabel('1/kappa')
        plt.ylabel('MSE')
        plt.grid()

        plt.show()

        plt.savefig('MSE_graph.png')