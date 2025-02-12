import numpy as np
from scipy.optimize import fsolve

# Here [CS18] is the reference Sur, Pragya, and Emmanuel J. Candès. "A modern
# maximum-likelihood theory for high-dimensional logistic regression." Proceedings
# of the National Academy of Sciences 116.29 (2019): 14516-14525.

class AuxiliaryFunctionsMLE:
    @staticmethod
    def rho_prime(t):
        return 1 / (1 + np.exp(-t))

    @staticmethod
    def rho_second(t):
        return 1 / (np.exp(-t / 2) + np.exp(t / 2))**2

    # Because we knwo explicitely the dereivarives of rho, we can compute
    # the value of prox_lambda(t) very fast by using Newton's method.
    def prox_newton(self, z, landa, max_iter=5, tolerance=1e-8):
        value = np.copy(z) 
        i, step = 0, 1.0
        while i < max_iter and np.max(np.abs(step)) > tolerance:
            i += 1
            f  = value - z + landa * self.rho_prime(value)
            df = 1 + landa * self.rho_second(value)
            step = f / df
            value_new = value - step
            value = value_new
        return value

class LogisticMeanFieldMLE(AuxiliaryFunctionsMLE):
    def __init__(self, kappa = 0.2, gamma = 1.0, n_samples = 1000000):
        self.kappa, self.gamma, self.n_samples = kappa, gamma, n_samples
        self.Z1 = np.random.normal(0.0,1.0,n_samples)
        self.Z2 = np.random.normal(0.0,1.0,n_samples)

    # Square root of the covairance matrix of (Q1,Q2). Used to compute Q1 and Q2
    # in terms of self.Z1 and self.Z2.
    def sqrt_covariance_Q(self,alpha,sigma):
        Sigma = np.zeros((2, 2))
        Sigma[0, 0] = self.gamma**2
        Sigma[0, 1] = Sigma[1,0] = - alpha*self.gamma**2
        Sigma[1, 1] = alpha**2 * self.gamma**2 + self.kappa* sigma**2
        evalues, evectors = np.linalg.eig(Sigma)
        assert (evalues >= 0).all()
        sqrt_Sigma = evectors * np.sqrt(evalues) @ np.linalg.inv(evectors)
        return sqrt_Sigma

    def compute_Q(self,alpha,sigma):
        sqrt_covariance_Q = self.sqrt_covariance_Q(alpha,sigma)
        Q1 = self.Z1 * sqrt_covariance_Q[0,0] + self.Z2 * sqrt_covariance_Q[0,1]
        Q2 = self.Z1 * sqrt_covariance_Q[1,0] + self.Z2 * sqrt_covariance_Q[1,1]
        return Q1, Q2

    def equation_1(self,sigma,landa,Q1,prox_Q2):
        right_hand_side = np.mean(np.multiply(self.rho_prime(Q1),self.rho_prime(prox_Q2)**2))
        left_hand_side = 0.5 * self.kappa**2 * sigma**2 / landa**2
        return right_hand_side - left_hand_side

    def equation_2(self,landa,Q1,prox_Q2):
        vector_samples = np.multiply(Q1,self.rho_prime(Q1))
        vector_samples = np.multiply(vector_samples,self.rho_prime(prox_Q2))
        right_hand_side = landa * np.mean(vector_samples)
        return  right_hand_side

    def equation_3(self,landa,Q1,prox_Q2):
        vector_1 = self.rho_prime(Q1)
        vector_2 = 1 / (1 + landa * self.rho_second(prox_Q2))
        right_hand_side = np.mean(np.multiply(vector_1,vector_2))
        left_hand_side = 0.5 * (1 - self.kappa)
        return right_hand_side - left_hand_side

    def differences_equations(self,sigma,landa,Q1,prox_Q2):
        return np.array([self.equation_1(sigma,landa,Q1,prox_Q2),
                        self.equation_2(landa,Q1,prox_Q2),
                        self.equation_3(landa,Q1,prox_Q2)])

    def loss_equations(self,alpha,sigma,landa):
            Q1, Q2 = self.compute_Q(alpha,sigma)
            prox_Q2 = self.prox_newton(Q2,landa)
            differences = self.differences_equations(sigma,landa,Q1,prox_Q2)
            print("Value of the loss: ",np.max(np.abs(differences)))
            return differences

    def find_solutions(self,alpha_0 = 2.0,sigma_0 = 20.0, landa_0 = 2.0,tolerance=1e-14):
        initial_values = np.array([alpha_0,sigma_0,landa_0])
        solution = fsolve(lambda X: self.loss_equations(X[0], X[1], X[2]),initial_values,xtol=tolerance,full_output=True)
        print("Output of optimisation: ",solution)
        return solution[0]