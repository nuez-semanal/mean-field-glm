import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mean_field_glm as mf

# prior_sigma fixed to 1.0

def main():
    block_arguments = {"variable": "snr",
                       "fixed_var": 1.0,
                       "num_per_var": 1,
                       #"var_tuple": (0.5,0.6,0.7,0.8,0.9,1.0,1.5,2.0,2.5,3.0,5.0,7.0,9.0,11.0,13.0,15.0),
                       "var_tuple": (0.5, 2.0, 5.0, 15.0),
                       "signal": "Rademacher",
                       "log_likelihood": "Logistic",
                       "bayes_optimal": False,
                       "file_name": "order_parameters_vs_snr_fixed_sigma_gaussian_rademacher"}

    block = mf.block_gaussian_computation.BlockGaussianComputation(**block_arguments)
    block.compute_data()

if __name__ == "__main__":
    main()
