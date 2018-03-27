import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random


example1 = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [4, 5], [4,6], [4,7], [5,6], [5,7], [6,7], [3,4]]
example1 = [[str(edge[0]), str(edge[1])] for edge in example1]

example2 = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [4, 5], [4,6], [4,7], [5,6], [5,7], [6,7]]
example2 = [[str(edge[0]), str(edge[1])] for edge in example2]

example3 = [[0, 1], [0, 2], [0, 3], [0, 4], [5, 6], [5, 7], [5, 8], [5, 9], [0, 5]]
example3 = [[str(edge[0]), str(edge[1])] for edge in example3]




def show_example_graph(g=False):
    if g is False:
        g = nx.Graph()
        edges = example1
        g.add_edges_from(edges)
    plt.figure()
    nx.draw(g, with_labels=True)
    plt.show()

def grad(g, E, node, nb_counts):
    N = g.number_of_nodes()

    grad = 0.0
    for u in range(N):
        dot = np.dot(E[node, :], E[u, :])
        if g.has_edge(str(node), str(u)):
            grad += -dot + nb_counts[node, u]*np.log(dot)
        else:
            grad += -dot

    return grad




def find_nb_counts(g):
    N = g.number_of_nodes()

    counts = np.zeros(shape=(N, N), dtype=np.int)
    for v in range(N):
        for u in range(N):
            pri
            v_nb = set(list(nx.neighbors(g, str(v))).append(str(u)))
            u_nb = set(list(nx.neighbors(g, str(u))).append(str(v)))
            counts[v, u] += len(v_nb.intersection(u_nb))

    return counts

def run():
    g = nx.Graph()
    edges = example1
    g.add_edges_from(edges)

    N = g.number_of_nodes()
    dim = 2
    eta = 0.001

    # Initialize
    E = np.random.normal(size=(N, dim))

    nb_counts = find_nb_counts(g)

    num_of_iters = 1
    for iter in range(num_of_iters):
        for node in range(N):
            grad_node = grad(g, E, node, nb_counts)
            E[node, :] += eta*grad_node

#show_example_graph()
run()