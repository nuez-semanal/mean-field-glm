import numpy as np

class AuxiliaryFunctions:
    @staticmethod
    def indicator(x):
        positive = x > 0
        result = np.zeros_like(x)
        result[positive] = 1
        return result

    @staticmethod
    def logit(x):
        return np.log(x/(1-x))

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def dif_sigmoid(x):
        return np.exp(-x) / (1 + np.exp(-x))**2

    @staticmethod
    def gauss_density(x):
        return np.exp(-0.5*x**2)/np.sqrt(2*np.pi)

    @staticmethod
    def tilde_t_delta(x,delta):
        return 0.5*(np.tanh(x/delta)+1.0)

    @staticmethod
    def dif_tilde_t_delta(x,delta):
        return 0.5*(1-np.tanh(x/delta)**2)/delta