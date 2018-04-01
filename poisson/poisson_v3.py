import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import *
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
            for nb_nb in nx.neighbors(g, nb):
                if int(nb_nb) not in nb_list[int(node)]:
                    nb_list[int(node)].append(int(nb_nb))
                for nb_nb_nb in nx.neighbors(g, nb_nb):
                    if int(nb_nb_nb) not in nb_list[int(node)]:
                        nb_list[int(node)].append(int(nb_nb_nb))

        if int(node) in nb_list[int(node)]:
            nb_list[int(node)].remove(int(node))

    return nb_list

def neighbors2(g, node):
    nb = []
    for v in nx.neighbors(g, node):
        if v not in nb:
            nb.append(v)
        for u in nx.neighbors(g, v):
            if u not in nb:
                nb.append(u)
    return nb

def find_common_nb(g):
    N = g.number_of_nodes()

    nbnb = lil_matrix(np.zeros(shape=(N, N), dtype=np.int), dtype=np.int)
    for v in g.nodes():
        for u in nx.neighbors(g, v):
            for w in nx.neighbors(g, u):
                if nbnb[int(v), int(w)] == 0:
                    v_nb = set(list(nx.neighbors(g, v)))
                    w_nb = set(list(nx.neighbors(g, w)))
                    intersect = v_nb.intersection(w_nb)
                    nbnb[int(v), int(w)] = len(intersect)
                for y in nx.neighbors(g, w):
                    if nbnb[int(v), int(y)] == 0:
                        v_nb = set(list(neighbors2(g, v)))
                        y_nb = set(list(neighbors2(g, y)))
                        intersect = v_nb.intersection(y_nb)
                        nbnb[int(v), int(y)] = len(intersect)

    return nbnb

def find_neighbors2(g):
    N = g.number_of_nodes()
    nb_list = [[] for _ in range(N)]

    for node in g.nodes():
        for nb in nx.neighbors(g, node):
            if int(nb) not in nb_list[int(node)]:
                #nb_list[int(node)].append(int(nb))
                for nb_nb in nx.neighbors(g, nb):
                    if int(nb_nb) not in nb_list[int(node)]:
                        nb_list[int(node)].append(int(nb_nb))
                        for nb_nb_nb in nx.neighbors(g, nb_nb):
                            if int(nb_nb_nb) not in nb_list[int(node)]:
                                nb_list[int(node)].append(int(nb_nb_nb))

        if int(node) in nb_list[int(node)]:
            nb_list[int(node)].remove(int(node))

    return nb_list


def find_common_nb2(g):
    N = g.number_of_nodes()

    nbnb = lil_matrix(np.zeros(shape=(N, N), dtype=np.int), dtype=np.int)
    for v in g.nodes():
        for u in nx.neighbors(g, v):
            for w in nx.neighbors(g, u):
                for x in nx.neighbors(g, w):
                    if nbnb[int(v), int(x)] == 0:
                        v_nb = set(list([j for i in nx.neighbors(g, v) for j in nx.neighbors(g, i)]))
                        x_nb = set(list([j for i in nx.neighbors(g, x) for j in nx.neighbors(g, i)]))
                        intersect = v_nb.intersection(x_nb)
                        nbnb[int(v), int(x)] = len(intersect)

    return nbnb

def grad(g, F, nb_list, common_nb, node):
    N = g.number_of_nodes()

    grad_sum = 0.0

    for u in range(N):
        N = float(common_nb[node, u])
        if u != node:
            grad_sum += -F[u, :]
            if N != 0:
                grad_sum += N * (F[u, :] / np.dot(F[node, :], F[u, :]))

    return grad_sum/2.0


def compute_score(g, F, nb_list, common_nb):

    N = g.number_of_nodes()


    score = 0.0
    for v in range(N):
        for u in range(v+1, N):
            Nvu = common_nb[v, u]
            dot = np.dot(F[v, :], F[u, :])
            score += -dot
            if Nvu != 0:
                score += float(Nvu)*np.log(dot + 1)
                score += -np.sum([np.log(val) for val in range(2, Nvu+1)])

    return score

def draw_points(F, name="", g=None, base=False):
    if F.shape[1] != 2:
        raise ValueError("Dim must be 2")


    if name == "Karate":
        groundTruth = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 1, 1, 1]
        plt.figure()
        for inx in range(g.number_of_nodes()):
            if groundTruth[inx] == 0:
                plt.plot(F[inx, 0], F[inx, 1], 'r.')
            if groundTruth[inx] == 1:
                plt.plot(F[inx, 0], F[inx, 1], 'b.')
        plt.show()

    else:
        plt.figure()
        plt.plot(T[:4, 0], T[:4, 1], 'b.')
        plt.plot(T[4:, 0], T[4:, 1], 'r.')
        plt.show()


def run(g, dim, num_of_iters, eta):
    N = nx.number_of_nodes(g)

    # Initialize parameters
    F = np.random.normal(size=(N, dim))
    F = np.absolute(F)

    #nb_list = find_neighbors(g)
    nb_list = []
    common_nb = find_common_nb(g)

    print("Iteration just started")
    for iter in range(num_of_iters):
        #if iter % 10 == 0:
        #    draw_points(B, T, "Karate", g, base=True)
        for node in range(N):
            node_grad = grad(g, F, nb_list, common_nb, node)
            F[node, :] += eta*node_grad
            F[node, :] = np.absolute(F[node, :])

        score = compute_score(g, F, nb_list, common_nb)
        print("Iter: {} Score {}".format(iter, score))

        if iter % 50 == 0:
            np.save("../comm/numpy_files/citeseer_poisson_v3_iter_{}".format(iter), F)
    return F


edges = example1
#g = nx.Graph()
#g.add_edges_from(edges)

g = nx.read_gml("../datasets/citeseer.gml")


F = run(g, dim=128, num_of_iters=300, eta=0.005)
np.save("../comm/numpy_files/citeseer_poisson_v3_son", F)
#draw_points(F, "Karate", g)
