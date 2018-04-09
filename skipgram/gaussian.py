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




def find_neighbors(g):
    N = g.number_of_nodes()
    nb_list = [[] for _ in range(N)]

    for node in g.nodes():

        for nb in nx.neighbors(g, node):
            if int(nb) not in nb_list[int(node)]:
                nb_list[int(node)].append(int(nb))

    return nb_list


def grad_T(g, B, T, node, v):
    N = g.number_of_nodes()

    normalizer = 1.0

    grad_sum = 0.0
    grad_sum += -(T[node, :] - B[v, :]) / normalizer

    return grad_sum

def grad_B(g, B, T, node, u):
    N = g.number_of_nodes()

    normalizer = 1.0

    grad_sum = 0.0
    grad_sum += +(T[u, :] - B[node, :]) / normalizer

    return grad_sum

def compute_score(g, B, T ,nb_list):

    N = g.number_of_nodes()

    normalizer = 2.0

    score = 0.0
    for v in range(N):
        for u in nb_list[v]:
            score += -np.dot(T[u, :] - B[v, :], T[u, :] - B[v, :]) / normalizer

    return score

def draw_points(B, T, name="", g=None):
    if B.shape[1] != 2:
        raise ValueError("Dim must be 2")


    if name == "Karate":
        groundTruth = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 1, 1, 1]
        plt.figure()
        for inx in range(g.number_of_nodes()):
            if groundTruth[inx] == 0:
                plt.plot(T[inx, 0], T[inx, 1], 'r.')
            if groundTruth[inx] == 1:
                plt.plot(T[inx, 0], T[inx, 1], 'b.')
        plt.show()

    else:
        plt.figure()
        plt.plot(T[:4, 0], T[:4, 1], 'b.')
        plt.plot(T[4:, 0], T[4:, 1], 'r.')
        plt.show()


def run(g, dim, num_of_iters, eta):
    N = nx.number_of_nodes(g)

    # Initialize parameters
    B = np.random.normal(size=(N, dim))
    T = np.random.normal(size=(N, dim))

    nb_list = find_neighbors(g)

    for iter in range(num_of_iters):
        nodes = range(N)
        np.random.shuffle(nodes)
        for node in nodes:



            for nb in nb_list[node]:
                node_grad_B = grad_T(g, B, T, node, nb)
                nb_grad_T = grad_T(g, B, T, nb, node)

                T[nb, :] += eta * nb_grad_T
                B[node, :] += eta * node_grad_B



        score = compute_score(g, B, T, nb_list)
        print("Iter: {} Score {}".format(iter, score))
        #np.save("./citeseer_F_iter_{}".format(iter), F)

    return B, T


edges = example1
#g = nx.Graph()
#g.add_edges_from(edges)
g = nx.read_gml("../datasets/karate.gml")

B, T = run(g, dim=2, num_of_iters=1050, eta=0.1)
#np.save("./numpy_files/citeseer_gaussian_final", T)
draw_points(B, T, "Karate", g)