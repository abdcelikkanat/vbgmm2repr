import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


F = np.load("./numpy_files/coeff_iter_99.npy")


g = nx.read_gml("../datasets/citeseer.gml")
cluster = nx.get_node_attributes(g, "clusters")


x = TSNE(n_components=2).fit_transform(F)


print(x.shape)

y = [[] for _ in range(6)]


for node in g.nodes():
    y[cluster[node]].append(x[int(node), :])


colors = ['r', 'b', 'g', 'y', 'c', 'm']
plt.figure()
for i in range(6):
    y[i] = np.asarray(y[i])
    plt.plot(y[i][:, 0], y[i][:, 1], ".{}".format(colors[i]))
plt.show()