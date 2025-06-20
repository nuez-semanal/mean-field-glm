import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mean_field_glm as mf

# prior_sigma takes value snr + 3.0

def main():
    block_arguments = {"variable": "snr",
                       "fixed_var": 0.25,
                       "num_per_var": 5,
                       "var_tuple": (0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0),
                       "signal": "Rademacher",
                       "log_likelihood": "Logistic",
                       "file_name": "comparison_mle_snr_variable_sigma_gaussian",
                       "bayes_optimal": True}

    block = mf.block_gaussian_computation.BlockGaussianComputation(**block_arguments)
    block.compute_data()

if __name__ == "__main__":
    main()
