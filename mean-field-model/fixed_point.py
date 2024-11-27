import numpy as np
from auxiliary import AuxiliaryFunctions
from probabilistic_model import ProbabilisticModel

class FixedPointEquations(AuxiliaryFunctions):
    def __init__(self,mean_field_model,v_b,c_b,c_bbs,r_1,r_2,r_3,n_iterations,tune,chains):
        self.mean_field_model = mean_field_model

        self.v_b, self.c_b, self.c_bbs = v_b, c_b, c_bbs
        self.r_1, self.r_2, self.r_3 = r_1, r_2, r_3
        self.t_gamma = self.compute_t_gamma()

        self.probabilistic_model = ProbabilisticModel(self,n_iterations,tune,chains)

    def get_mean_field_parameters(self):
        n = self.mean_field_model.n
        p = self.mean_field_model.p
        delta = self.mean_field_model.delta
        snr = self.mean_field_model.snr
        prior = self.mean_field_model.prior
        param1 = self.mean_field_model.param1
        param2 = self.mean_field_model.param2
        return n, p, delta, snr, prior, param1, param2

    def compute_t_gamma(self):
        n, p, delta = self.get_mean_field_parameters()[:3]

        kappa = p/n
        d = delta

        range_xth = np.linspace(-5,5,1000)
        X = np.array([[[x_th,x_e] for x_e in np.linspace(kappa*x_th-5*d,kappa*x_th+5*d,1000)] for x_th in range_xth])
        slice_integrals = []

        for i in range(1000):
            th = X[i,:][0,0]
            e = X[i,:][:,1]
            y = self.gauss_density(th)*np.multiply(self.dif_tilde_t_delta(kappa*th-e),self.dif_sigmoid(e))
            slice_integrals.append(np.trapz(y,e))

        return np.trapz(slice_integrals,range_xth)

    def compute_s_theta(self,observations,posterior):
        delta = self.get_mean_field_parameters()[2]
        return self.tilde_t_delta(observations,delta)-self.sigmoid(posterior)

    def compute_s_theta_star(self,observations,posterior):
        delta = self.get_mean_field_parameters()[2]
        return np.multiply(self.dif_tilde_t_delta(observations,delta),posterior)

    def update_order_parameters1(self):
        v_b, c_b, c_bbs = 0,0,0
        p = self.mean_field_model.p
        posterior = self.probabilistic_model.posterior1
        true_beta = self.probabilistic_model.true_beta
        chains = self.probabilistic_model.chains
        normalisation = chains * p

        for i in range(chains):
            v_b += np.sum(posterior[i]**2)/normalisation
            t, s = np.random.randint(0,chains,2)
            c_b += np.dot(posterior[t],posterior[s])/normalisation
            c_bbs += np.dot(posterior[i],true_beta)/normalisation

        self.v_b, self.c_b, self.c_bbs = v_b, c_b, c_bbs

    def update_order_parameters2(self):

        tilde_v, tilde_q, tilde_m, bar_m, t_gamma, a_dp = 0,0,0,0,0,0
        n = self.mean_field_model.n
        posterior = self.probabilistic_model.posterior2
        observations = self.probabilistic_model.observations2
        chains = self.probabilistic_model.chains
        normalisation = chains * n

        for i in range(chains):

            s_theta = self.compute_s_theta(observations, posterior[i])
            tilde_v += np.dot(s_theta,s_theta)/normalisation

            t, s = np.random.randint(0,chains,2)
            s_theta_1 = self.compute_s_theta(observations, posterior[t])
            s_theta_2 = self.compute_s_theta(observations, posterior[s])
            tilde_q += np.dot(s_theta_1,s_theta_2)/normalisation

            s_theta_star = self.compute_s_theta_star(observations, posterior[i])
            s_theta = self.compute_s_theta(observations, posterior[i])
            tilde_m += np.dot(s_theta_star,s_theta)/normalisation

            s_theta_star_1 = self.compute_s_theta_star(observations, posterior[t])
            s_theta_2 = self.compute_s_theta(observations, posterior[s])
            bar_m += np.dot(s_theta_star_1,s_theta_2)/normalisation

            a_dp += np.sum(self.dif_sigmoid(posterior[i]))/normalisation

        r_1 = a_dp + tilde_q - tilde_v
        r_2 = t_gamma + tilde_m - bar_m
        r_3 = np.sqrt(tilde_q)

        self.r_1, self.r_2, self.r_3 = r_1, r_2, r_3

    def return_order_parameters(self):
        return self.v_b, self.c_b, self.c_bbs, self.r_1, self.r_2, self.r_3

    def run_one_iteration(self):
        self.probabilistic_model.draw_samples()
        self.update_order_parameters1()
        self.update_order_parameters2()
        self.probabilistic_model.update_mean_field_model1()
        self.probabilistic_model.update_mean_field_model2()