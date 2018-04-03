import numpy as np
import networkx as nx


<<<<<<< HEAD
F = np.load("./numpy_files/citeseer_mcmc_T.npy")
=======
F = np.load("./numpy_files/citeseer_gaussian_v3_iter_150.npy")
>>>>>>> b1faaaaf6a302f9c21c2fdc982fdd6037c0b51c6

g = nx.read_gml("../datasets/citeseer.gml")

n = g.number_of_nodes()

<<<<<<< HEAD
with open("citeseer_mcmc_T.embedding", 'w') as f:
=======
with open("citeseer_gaussian_v3_iter_150.embedding", 'w') as f:
>>>>>>> b1faaaaf6a302f9c21c2fdc982fdd6037c0b51c6
    f.write("{} {}\n".format(n, F.shape[1]))
    for node in g.nodes():
        line = [str(val) for val in F[int(node), :]]

        f.write("{} {}\n".format(node, " ".join(line)))