import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mean_field_model.graphs import MseGraphCreator


def main():
    graph_arguments = {"variable": "kappa",
                       "fixed_var": 1.0,
                       "num_per_var": 10,
                       "var_tuple": (5.0, 4.0, 3.0, 2.0, 1.0, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 6),
                       "prior": "Normal",
                       "signal": "Normal",
                       "log_likelihood": "Logistic",
                       "file_name": "order_parameters_vs_kappa",
                       "bayes_optimal": True}

    graph_creator = MseGraphCreator(**graph_arguments)

    data = np.loadtxt("order_parameters_vs_kappa.csv",delimiter=",")

    data[:,1] = 1/data[:,1]

    graph_creator.data = data

    graph_creator.plot_graph_mse(save=True,limits=[0.8,0.935])

    print(graph_creator.data)

    print(graph_creator.stats)

if __name__ == "__main__":
    main()