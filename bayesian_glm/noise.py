import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mean_field_glm.auxiliary import AuxiliaryFunctions

class NoiseComputer(AuxiliaryFunctions):
    def __init__(self, model, hq ):
        self.model = model
        self.hq = hq

    def cavity_fitted_values( self, k, M ):
        cavity_indices = (np.arange(self.model.p) != k)
        return np.dot(self.model.data[:,cavity_indices],self.model.posterior[:M,cavity_indices].transpose())

    def cavity_observations( self, k ):
        cavity_indices = (np.arange(self.model.p) != k)
        cavity_projections = np.dot(self.model.data[:,cavity_indices],self.model.true_beta[cavity_indices])

        if self.model.log_likelihood == "Logistic":
            cavity_obs = self.indicator(np.sqrt(self.model.snr)*cavity_projections-self.model.noise)
        else:
            cavity_obs = np.sqrt(self.model.snr)*cavity_projections-self.model.noise

        return cavity_obs

    def compute_theta( self, k, j ):
        fitted_values = self.cavity_fitted_values(k,j)
        observations = self.cavity_observations(k)
        pre_theta = np.zeros_like(fitted_values)

        if self.model.log_likelihood == "Logistic":
            for i in range(j):
                pre_theta[:,i] = observations - self.sigmoid(np.sqrt(self.model.snr)*fitted_values[:,i])
        else:
            for i in range(j):
                pre_theta[:,i] = observations - np.sqrt(self.model.snr)*fitted_values[:,i]

        return np.matmul(self.model.data[:,k],pre_theta)

    def compute_noise( self, k, j = 100 ):
        theta = self.compute_theta( k, j )
        return np.mean(theta) / np.sqrt(self.hq)