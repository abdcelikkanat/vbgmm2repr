import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def get_nb_list(g):
    return [[int(nb) for nb in nx.neighbors(g, str(node))] for node in range(g.number_of_nodes())]

example1 = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [4, 5], [4,6], [4,7], [5,6], [5,7], [6,7], [3,4]]
example1 = [[str(edge[0]), str(edge[1])] for edge in example1]


edges = example1


g = nx.Graph()
g.add_edges_from(edges)

#g = nx.read_gml("../datasets/karate.gml")

"""
plt.figure()
nx.draw(g)
plt.show()
"""

N = g.number_of_nodes()
D = 1
K = 2
eta = 0.01

F = np.random.normal(size=(N, K, D))*1.0
nb_list = get_nb_list(g)

F = np.abs(F)

epsilon = (2.0*float(g.number_of_edges())) / (float(N) * (float(N)-1.0))
print(epsilon)

num_of_iters = 1

def one_iter(B, T, cluster, eta):

    for node in range(N):
        delta_T = 0.0
        for s in range(N):
            delta_T += (cluster[node] == cluster[s]) * (B[s, :] / (1.0 - np.exp(-T[node, :] * B[s, :]))) + \
                           (cluster[node] != cluster[s]) * (-B[s, :])

        B = B + eta*delta_T

        delta_B = 0.0
        for t in range(N):
            delta_B += (cluster[t] == cluster[node]) * (T[t, :] / (1.0 - np.exp(-T[t, :] * B[node, :]))) + \
                          (cluster[t] != cluster[node]) * (-T[node, :])

        T = T + eta*delta_B

def detect_clusters(g):
    n = g.number_of_nodes()



    for node in range(n):
        clusters

def run():
    dim = 2
    num_of_iters = 1

    # Initialize
    B = np.random.normal(size=dim)
    T = np.random.normal(size=dim)

    for iter in range(num_of_iters):
        one_iter(B, T )

