import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mean_field_glm.graphs import MarginalGraphCreator

# Values of parameters fixed by results of simulations from "compute_order_parameters_vs_kappa_beta.py"

parameters_simulations = [0.151888443457735,0.188428694761769,0.389989312324746]

def main():

    # Parameters correspond to (r1,r2,r3) for Beta(2,5) signal, Beta(2,2) prior, gamma = 10.0, and kappa = 1.0
    marginal_graph_creator = MarginalGraphCreator(
        n=1000,
        p=1000,
        gamma=1.0, # Now Model Beta code does not implement custom gamma. Fix always to 1.0.
        prior="Beta",
        signal="Beta",
        parameters=parameters_simulations
    )

    marginal_graph_creator.histogram_marginal()

    marginal_graph_creator.qq_plot_marginal()



if __name__ == "__main__":
    main()
