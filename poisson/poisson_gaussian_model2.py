import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

example1 = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [4, 5], [4, 6], [4, 7], [5, 6], [5, 7], [6, 7], [3, 4]]
example1 = [[str(edge[0]), str(edge[1])] for edge in example1]

example2 = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [4, 5], [4, 6], [4, 7], [5, 6], [5, 7], [6, 7]]
example2 = [[str(edge[0]), str(edge[1])] for edge in example2]

example3 = [[0, 1], [0, 2], [0, 3], [0, 4], [5, 6], [5, 7], [5, 8], [5, 9], [0, 5]]
example3 = [[str(edge[0]), str(edge[1])] for edge in example3]

"""
T(i) is the embedding of node i
B(k) is the embedding of the imaginary node k dedicated to the cluster k

e_{ij} ~ Bern(\pi_{ij})
\pi_{ij} = P(k > 0)
k ~ Poisson(\lambda_{ij})
\lambda_{ij} = \sum_k P(i|k)P(j|k)z_k^2 where z_k is the normalization factor
P(i|k) = N(T(i); B(k), I),
Therefore,
    P(i|k)z_k = exp(-(T(i)-B(k))^2/2)


"""


def find_neighbors(g):
    N = g.number_of_nodes()
    nb_list = [[] for _ in range(N)]

    for node in g.nodes():
        for nb in nx.neighbors(g, node):
            if int(nb) not in nb_list[int(node)]:
                nb_list[int(node)].append(int(nb))

    return nb_list


def find_neighbors3(g):
    N = g.number_of_nodes()
    nb_list = [[] for _ in range(N)]

    nb_list2 = [[] for _ in range(N)]

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


    return nb_list

def grad_lambda_node(T, B, node, u):
    K = B.shape[0]

    grad_sum = 0.0
    for k in range(K):
        term1 = np.exp(np.dot(T[node, :], B[k, :]) + np.dot(T[u, :], B[k, :]))
        term2 = B[k, :]
        grad_sum += term1 * term2

    return grad_sum


def grad_lambda_k(T, B, v, u, k):
    term1 = np.exp(np.dot(T[v, :], B[k, :]) + np.dot(T[u, :], B[k, :]))
    term2 = (T[v, :] + T[u, :])

    return term1 * term2


def grad_T(g, B, T, nb_list, lambda_matrix, node):
    N = g.number_of_nodes()
    K = B.shape[0]

    var = 1.0

    grad_sum = 0.0
    for u in range(N):
        grad_lambda = grad_lambda_node(T, B, node, u)
        if u in nb_list[node]:
            grad_sum += (1.0 / (np.exp(lambda_matrix[str(node)][str(u)]) - 1.0)) * grad_lambda
        else:
            grad_sum += -grad_lambda

    return grad_sum


def grad_B(g, B, T, nb_list, lambda_matrix, node_k):
    N = g.number_of_nodes()

    var = 1.0

    grad_sum = 0.0
    for v in range(N):
        for u in range(N):
            grad_k = grad_lambda_k(T, B, v, u, node_k)
            if u in nb_list[v]:
                grad_sum += (1.0 / (np.exp(lambda_matrix[str(v)][str(u)]) - 1.0)) * grad_k
            else:
                grad_sum += -grad_k

    return grad_sum


def compute_lambda(g, T, B, nb_list):
    N = nx.number_of_nodes(g)
    K = B.shape[0]

    lambda_matrix = {str(v): {str(u): 0.0 for u in range(N)} for v in range(N)}

    for v in range(N):
        for u in range(N):
            if lambda_matrix[str(v)][str(u)] == 0.0 and u != v:

                sum_k = 0.0
                for k in range(K):
                    sum_k += np.exp(np.dot(T[v, :], B[k, :]) + np.dot(T[u, :], B[k, :]))

                lambda_matrix[str(v)][str(u)] = sum_k
                lambda_matrix[str(u)][str(v)] = sum_k

    return lambda_matrix


def compute_score(g, nb_list, lambda_matrix):
    N = g.number_of_nodes()

    var = 1.0

    score = 0.0
    for v in range(N):
        for u in range(N):

            if u in nb_list[v]:
                score += np.log(1.0 - np.exp(-lambda_matrix[str(v)][str(u)]))
            else:
                if v != u:
                    score += -lambda_matrix[str(v)][str(u)]

    return score


def draw_points(B, T, name="", g=None, base=False):
    if B.shape[1] != 2:
        raise ValueError("Dim must be 2")

    if name == "Karate":
        groundTruth = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 1, 1, 1]
        plt.figure()
        for inx in range(g.number_of_nodes()):
            if groundTruth[inx] == 0:
                plt.plot(T[inx, 0], T[inx, 1], 'r.')
                # if base is True:
                #    plt.plot(B[inx, 0], B[inx, 1], 'rx')
            if groundTruth[inx] == 1:
                plt.plot(T[inx, 0], T[inx, 1], 'b.')
                # if base is True:
                #    plt.plot(B[inx, 0], B[inx, 1], 'bx')

        if base is True:
            plt.plot(B[:, 0], B[:, 1], 'gx')

        plt.show()

    else:
        plt.figure()
        plt.plot(T[:4, 0], T[:4, 1], 'b.')
        plt.plot(T[4:, 0], T[4:, 1], 'r.')
        plt.show()


def run(g, dim, num_of_iters, eta, num_of_classes):
    N = nx.number_of_nodes(g)

    # Initialize parameters
    B = np.random.normal(size=(num_of_classes, dim))
    T = np.random.normal(size=(N, dim))


    nb_list = find_neighbors3(g)

    # dist = find_distances(g)
    dist = []

    for iter in range(num_of_iters):
        # if iter % 10 == 0:
        #    draw_points(B, T, "Karate", g, base=True)
        for k in range(num_of_classes):
            lambda_matrix = compute_lambda(g, T, B, nb_list)

            node_grad_k = grad_B(g, B, T, nb_list, lambda_matrix, node_k=k)
            B[k, :] += eta * node_grad_k

        node_grad_T = np.zeros(shape=(N, dim), dtype=np.float)
        for node in range(N):
            lambda_matrix = compute_lambda(g, T, B, nb_list)

            node_grad_T[node, :] = grad_T(g, B, T, nb_list, lambda_matrix, node)
            T[node, :] += eta * node_grad_T[node, :]

        # for node in range(N):
        #    T[node, :] += eta * node_grad_T[node]
        lambda_matrix = compute_lambda(g, T, B, nb_list)
        score = compute_score(g, nb_list, lambda_matrix)
        print("Iter: {} Score {}".format(iter, score))

        # if iter % 50 == 0:
        #    np.save("./numpy_files/citeseer_poisson_gaussian_iter_{}".format(iter), T)
    # draw_points(B, T, "Karate", g, base=False)

    return B, T


# edges = example1
# g = nx.Graph()
# g.add_edges_from(edges)
g = nx.read_gml("../datasets/karate.gml")

B, T = run(g, dim=2, num_of_iters=1000, eta=0.000001, num_of_classes=2)
# np.save("./numpy_files/citeseer_poisson_gaussian_iter_son", T)
draw_points(B, T, "Karate", g, base=True)

