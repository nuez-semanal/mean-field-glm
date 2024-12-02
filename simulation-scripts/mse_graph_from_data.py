import sys
import os
import json
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mean_field_glm.graphs import MseGraphCreator


def main():

    file_name = input("Name of csv file without extension: ")

    file_path = "./simulation-results/" + file_name + '.json'

    with open(file_path, 'r') as file:
        graph_arguments = json.load(file)

    graph_creator = MseGraphCreator(**graph_arguments)

    data = np.loadtxt("./simulation-results/" + file_name + ".csv",delimiter=",")

    graph_creator.data = data

    graph_creator.plot_graph_mse(save=True)

if __name__ == "__main__":
    main()