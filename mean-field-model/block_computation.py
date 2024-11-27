import numpy as np
from IPython.display import clear_output
from mean_field import MeanFieldModel

class BlockComputation:
    def __init__(self,kappa_list = [1.0], num_per_kappa = 5, delta = 1.0, snr = 1.0, prior = "Normal",
                 signal = "Normal",noise=True,approx_obs=False):
        self.prior, self.signal = prior, signal
        self.kappa_list, self.num_per_kappa, self.delta = kappa_list, num_per_kappa, delta
        self.snr, self.noise, self.approx_obs = snr, noise, approx_obs
        self.data = None

    def compute_data(self):
        data = np.zeros(shape=(len(self.kappa_list)*self.num_per_kappa,7))
        for i in range(len(self.kappa_list)):
            for n in range(self.num_per_kappa):
                k = self.kappa_list[i]
                print("%%%%%%%%%% COMPUTING DATA POINT",n+1,"OUT OF",self.num_per_kappa,"FOR KAPPA =",k,"%%%%%%%%%%\n\n")
                model = MeanFieldModel(p = 10000, n = int(10000/k), delta = self.delta, prior = self.prior,
                                       signal = self.signal, snr = self.snr)
                model.run_iterations()

                parameters = model.show_order_parameters()

                data[n+self.num_per_kappa*i,0] = k
                for j in range(6):
                    data[n+self.num_per_kappa*i,j+1] = parameters[j]
                clear_output(wait=True)

        self.data = data

    def check_if_data(self):
        if self.data is None:
            print("Please, compute the data first!")
            raise ValueError()

    def save_data(self,name="Computed data"):
        self.check_if_data()
        np.savetxt(name+".csv", self.data, delimiter=",")