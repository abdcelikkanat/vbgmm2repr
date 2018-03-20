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

    nb_counts = np.zeros(shape=(N, N), dtype=np.int)
    for v in range(N):
        for u in range(v+1, N):
            nb_v = list(nx.neighbors(g, str(v)))
            nb_v.append('u')
            nb_u = list(nx.neighbors(g, str(u)))
            nb_u.append('v')

            inter = set(nb_v).intersection(nb_u)

            nb_counts[v, u] = len(inter)
            nb_counts[u, v] = nb_counts[v, u]

        nb_counts[v, v] = 0

    return nb_counts


def grad(g, F, nb_counts, node):
    N = g.number_of_nodes()

    grad_sum = 0.0
    for u in range(N):
        k = nb_counts[node, u]
        dot = np.dot(F[node, :], F[u, :])
        if k != 0:
            grad_sum += F[u, :] * (-2.0*dot + ( ( 2.0 * k ) / (dot + 1e-8)) )
        else:
            grad_sum += -2.0*F[u, :] * dot

    return grad_sum

def compute_score(g, F ,nb_counts):

    N = g.number_of_nodes()

    score = 0.0
    for v in range(N):
        for u in range(v+1, N):
            k = nb_counts[v, u]
            dot = np.dot(F[v, :], F[u, :])
            if k != 0:
                score += -(dot*dot) + k*np.log(dot*dot +1e-8)
            else:
                score += -dot
    return score

def draw_points(F, name="", g=None):
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
        plt.plot(F[:4, 0], F[:4, 1], 'b.')
        plt.plot(F[4:, 0], F[4:, 1], 'r.')
        plt.show()


def run(g, dim, num_of_iters, eta):
    N = nx.number_of_nodes(g)

    # Initialize parameters
    F = np.random.normal(size=(N, dim))
    #F = np.absolute(F)
    nb_counts = find_neighbors(g)

    for iter in range(num_of_iters):
        for node in range(N):
            node_grad = grad(g, F, nb_counts, node)

            F[node, :] += eta * node_grad

            #F[node, :] = ( F[node, :] + np.absolute(F[node, :]) ) / 2.0

        score = compute_score(g, F, nb_counts)
        print("Iter: {} Score {}".format(iter, score))
        #np.save("./numpy_files/v2_coeff_iter_{}".format(iter), F)

    return F


edges = example1
#g = nx.Graph()
#g.add_edges_from(edges)
g = nx.read_gml("../datasets/karate.gml")

F = run(g, dim=2, num_of_iters=350, eta=0.0001)

draw_points(F, "Karate", g)