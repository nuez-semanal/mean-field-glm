import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mean_field_glm as mf

# prior_sigma is equal to snr

def main():
    block_arguments = {"variable" : "kappa",
                       "fixed_var" : 4.0,
                       "num_per_var" : 5,
                       "var_tuple" : (0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5),
                       "signal" : "Normal",
                       "log_likelihood" : "Logistic",
                       "file_name" : "order_parameters_vs_kappa_gaussian"}

    block = mf.block_gaussian_computation.BlockGaussianComputation(**block_arguments)

    block.compute_data()

if __name__ == "__main__":
    main()