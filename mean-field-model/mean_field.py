import numpy as np
from fixed_point import FixedPointEquations

class MeanFieldModel:
    def __init__(self,p = 400,n = 1000, tolerance = 0.01, draws = 1000, chains = 4, tune = 1000, n_iterations = 5,v_b = 1.0,c_b = 0.0,c_bbs = 0.0,
                 r_1 = 1e-6, r_2 = 0.0, r_3 = 0.0,signal="Normal",prior = "Normal",snr=1.0, delta = 1.0,
                 param1 = 0.0,param2 = 1.0, seed = False):

        self.p, self.n, self.param1, self.param2, self.tolerance = p, n, param1, param2, tolerance
        self.signal, self.prior, self.snr, self.delta  = signal, prior, snr, delta

        self.n_iterations = n_iterations

        self.fixed_point_equations = FixedPointEquations(self,v_b,c_b,c_bbs,r_1,r_2,r_3,n_iterations,draws,tune,chains)

        if seed is False:
            seed = np.random.randint(1000)

        np.random.seed(seed)

    def run_iterations(self):
        i, distance = 0, 1.0

        while distance > self.tolerance and i < self.n_iterations:
            i += 1
            distance = self.check_stability()

            v_b, c_b, c_bbs, r_1, r_2, r_3 = self.fixed_point_equations.return_order_parameters()

            print(f"\n[{i} iteration/s out of {self.n_iterations} max ready. Distance achieved = {distance}]\n")
            print("Actual values: ",v_b,c_b,c_bbs,r_1,r_2,r_3,"\n")

    def check_stability(self):
        old_order_parameters = np.array(self.fixed_point_equations.return_order_parameters())
        self.fixed_point_equations.run_one_iteration()
        new_order_parameters = np.array(self.fixed_point_equations.return_order_parameters())
        return np.linalg.norm(old_order_parameters-new_order_parameters)/np.linalg.norm(old_order_parameters)

    def show_order_parameters(self):
        order_parameters = self.fixed_point_equations.return_order_parameters()
        print(order_parameters)
        return order_parameters