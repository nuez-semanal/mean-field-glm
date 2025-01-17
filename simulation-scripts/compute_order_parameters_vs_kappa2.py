import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mean_field_glm import block_computation

# 6/2,8/10,1.0

def main():
    block_arguments = {"variable": "kappa",
                       "fixed_var": 100.0,
                       "num_per_var": 20,
                       "var_tuple": (0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,3.0),
                       "prior": "Beta",
                       "signal": "Beta",
                       "log_likelihood": "Logistic",
                       "file_name": "order_parameters_vs_kappa2",
                       "bayes_optimal": False}

    block = block_computation.BlockComputation(**block_arguments)
    block.compute_data()

if __name__ == "__main__":
    main()