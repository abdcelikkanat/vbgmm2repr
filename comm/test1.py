import numpy as np
import networkx as nx


F = np.load("./numpy_files/citeseer_poisson_model1_iter_50.npy")

g = nx.read_gml("../datasets/citeseer.gml")

n = g.number_of_nodes()

with open("citeseer_poisson_model1_iter_50.embedding", 'w') as f:
    f.write("{} {}\n".format(n, F.shape[1]))
    for node in g.nodes():
        line = [str(val) for val in F[int(node), :]]

        f.write("{} {}\n".format(node, " ".join(line)))