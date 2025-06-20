import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mean_field_glm as mf

# prior_sigma is snr + 3.0

def main():
    block_arguments = {"variable": "kappa",
                       "fixed_var": 1.0,
                       "num_per_var": 5,
                       "var_tuple": (0.1,0.15,0.2,0.25,0.3,0.35,0.4),
                       "signal": "Rademacher",
                       "log_likelihood": "Logistic",
                       "file_name": "comparison_mle_kappa_gaussian_rademacher"}

    block = mf.block_gaussian_computation.BlockGaussianComputation(**block_arguments)
    block.compute_data()


if __name__ == "__main__":
    main()