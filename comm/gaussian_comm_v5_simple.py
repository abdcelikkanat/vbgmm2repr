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

example4 = [[0,1], [1,2], [0,2]]

m = 10
example5 = [[i, j] for i in range(m) for j in range(i+1, m)]

example6 = [[0,1], [1,2], [2,3], [3, 0]]


def find_distances(graph):
    N = nx.number_of_nodes(graph)

    spl = nx.shortest_path_length(graph)

    dist = np.zeros(shape=(N, N), dtype=np.int)

    for p in spl:
        for target in p[1]:
            dist[int(p[0]), int(target)] = p[1][target]

    return dist


def find_neighbors(g):
    N = g.number_of_nodes()
    nb_list = [[] for _ in range(N)]

    for node in g.nodes():
        for nb in nx.neighbors(g, node):
            if int(nb) not in nb_list[int(node)]:
                nb_list[int(node)].append(int(nb))
                for nb_nb in nx.neighbors(g, nb):
                    if int(nb_nb) not in nb_list[int(node)]:
                        nb_list[int(node)].append(int(nb_nb))


    return nb_list

def grad(g, E, nb_list, node, dist):
    N = g.number_of_nodes()

    var = 1.0

    grad_sum = 0.0
    for v in range(N):
        for u in nb_list[v]:
            if node == v:
                grad_sum += +(E[u, :] - E[node, :]) / var

            if node == u:
                grad_sum += -(E[node, :] - E[v, :]) / var

    return  grad_sum


def compute_score(g, E, nb_list, dist):

    N = g.number_of_nodes()

    var = 1.0
    #*float(dist[v, u])*float(dist[v, u])
    score = 0.0
    for v in range(N):
        for u in nb_list[v]:
            score += -np.dot(E[u, :] - E[v, :], E[u, :] - E[v, :]) / (var*2.0)

    return score

def draw_points(E, name="", g=None, base=False):
    if E.shape[1] != 2:
        raise ValueError("Dim must be 2")


    if name == "Karate":
        groundTruth = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 1, 1, 1]
        plt.figure()
        for inx in range(g.number_of_nodes()):
            if groundTruth[inx] == 0:
                plt.plot(E[inx, 0], E[inx, 1], 'r.')
            if groundTruth[inx] == 1:
                plt.plot(E[inx, 0], E[inx, 1], 'b.')
        plt.show()

    elif name == "Trio":

        plt.figure()
        plt.plot(E[:, 0], E[:, 1], 'r.')
        plt.show()

    else:
        plt.figure()
        plt.plot(T[:4, 0], T[:4, 1], 'b.')
        plt.plot(T[4:, 0], T[4:, 1], 'r.')
        plt.show()


def run(g, dim, num_of_iters, eta):
    N = nx.number_of_nodes(g)

    # Initialize parameters
    E = np.random.normal(size=(N, dim))


    nb_list = find_neighbors(g)

    #dist = find_distances(g)
    dist = []

    for iter in range(num_of_iters):
        if iter % 50 == 0:
            draw_points(E, "Karate", g, base=True)
        for node in range(N):

            node_grad_E = grad(g, E, nb_list, node, dist)

            E[node, :] += eta * node_grad_E

        score = compute_score(g, E, nb_list, dist)
        print("Iter: {} Score {}".format(iter, score))

    return E


edges = example5
#g = nx.Graph()
#g.add_edges_from(edges)
g = nx.read_gml("../datasets/karate.gml")


E = run(g, dim=2, num_of_iters=1000, eta=0.001)
#np.save("./numpy_files/citeseer_gaussian_v5", E)
draw_points(E, "Karate", g)