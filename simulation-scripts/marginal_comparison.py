from mean_field_glm.graphs import MarginalGraphCreator

def main():

    marginal_graph_creator = MarginalGraphCreator()

    marginal_graph_creator.histogram_marginal()

    marginal_graph_creator.qq_plot_marginal()

if __name__ == "__main__":
    main()