import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import *
import random
"""
F_i is the weight of the node i
N_ij is the number of common nodes between i and j: the s-neighbour N_ij can be changed?

N_ij ~ Poisson(\lambda_{ij})
\lambda_{ij} = <F_i, F_j> 

Result:
Karate dataset i icin nb2, nb1 ve nb2_v2 kullandigimda guzel goruntuler cikiyor ama diger durumlarda pek iyi degil gibi

Citeseer icin adam gibi converge etmeyen 50. iter de 31.0 basari orani elede edildi.

"""



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
                    for nb_nb_nb_nb in nx.neighbors(g, nb_nb_nb):
                        if int(nb_nb_nb_nb) not in nb_list[int(node)]:
                            nb_list[int(node)].append(int(nb_nb_nb_nb))


        if int(node) in nb_list[int(node)]:
            nb_list[int(node)].remove(int(node))

    return nb_list


def neighbors1(g):
    nb_list = {node: [] for node in g.nodes()}

    for node in g.nodes():
        for v in nx.neighbors(g, node):
            if v not in nb_list[node]:
                nb_list[node].append(v)

    return nb_list

def neighbors2(g):
    nb_list = {node: [] for node in g.nodes()}

    for node in g.nodes():
        for v in nx.neighbors(g, node):
            if v not in nb_list[node]:
                nb_list[node].append(v)
            for u in nx.neighbors(g, v):
                if u not in nb_list[node]:
                    nb_list[node].append(u)

    return nb_list

def neighbors3(g):
    nb_list = {node: [] for node in g.nodes()}

    for node in g.nodes():
        for v in nx.neighbors(g, node):
            if v not in nb_list[node]:
                nb_list[node].append(v)
            for u in nx.neighbors(g, v):
                if u not in nb_list[node]:
                    nb_list[node].append(u)
                for w in nx.neighbors(g, u):
                    if w not in nb_list[node]:
                        nb_list[node].append(w)
    return nb_list

def neighbors4(g):
    nb_list = {node: [] for node in g.nodes()}

    for node in g.nodes():
        for v in nx.neighbors(g, node):
            if v not in nb_list[node]:
                nb_list[node].append(v)
            for u in nx.neighbors(g, v):
                if u not in nb_list[node]:
                    nb_list[node].append(u)
                for w in nx.neighbors(g, u):
                    if w not in nb_list[node]:
                        nb_list[node].append(w)
                    for x in nx.neighbors(g, w):
                        if x not in nb_list[node]:
                            nb_list[node].append(x)

        if node in nb_list[node]:
            nb_list[node].remove(node)

    return nb_list

def neighbors5(g):
    nb_list = {node: [] for node in g.nodes()}

    for node in g.nodes():
        for v in nx.neighbors(g, node):
            if v not in nb_list[node]:
                nb_list[node].append(v)
            for u in nx.neighbors(g, v):
                if u not in nb_list[node]:
                    nb_list[node].append(u)
                for w in nx.neighbors(g, u):
                    if w not in nb_list[node]:
                        nb_list[node].append(w)
                    for x in nx.neighbors(g, w):
                        if x not in nb_list[node]:
                            nb_list[node].append(x)
                        for y in nx.neighbors(g, x):
                            if y not in nb_list[node]:
                                nb_list[node].append(y)

        if node in nb_list[node]:
            nb_list[node].remove(node)

    return nb_list

def neighbors6(g):
    nb_list = {node: [] for node in g.nodes()}

    for node in g.nodes():
        for v in nx.neighbors(g, node):
            if v not in nb_list[node]:
                nb_list[node].append(v)
            for u in nx.neighbors(g, v):
                if u not in nb_list[node]:
                    nb_list[node].append(u)
                for w in nx.neighbors(g, u):
                    if w not in nb_list[node]:
                        nb_list[node].append(w)
                    for x in nx.neighbors(g, w):
                        if x not in nb_list[node]:
                            nb_list[node].append(x)
                        for y in nx.neighbors(g, x):
                            if y not in nb_list[node]:
                                nb_list[node].append(y)
                            for z in nx.neighbors(g, y):
                                if z not in nb_list[node]:
                                    nb_list[node].append(z)

        if node in nb_list[node]:
            nb_list[node].remove(node)

    return nb_list

def find_common_nb11(g):
    N = g.number_of_nodes()

    nbnb = lil_matrix(np.zeros(shape=(N, N), dtype=np.int), dtype=np.int)
    for v1 in g.nodes():
        for v2 in nx.neighbors(g, v1):
            for v3 in nx.neighbors(g, v2):
                if nbnb[int(v1), int(v3)] == 0:
                    v_nb = set(list(nx.neighbors(g, v1)))
                    w_nb = set(list(nx.neighbors(g, v3)))
                    intersect = v_nb.intersection(w_nb)
                    nbnb[int(v1), int(v3)] = len(intersect)

    return nbnb

def find_common_nb22(g):
    N = g.number_of_nodes()

    nb1 = neighbors1(g)
    nb2 = neighbors2(g)

    nbnb = lil_matrix(np.zeros(shape=(N, N), dtype=np.int), dtype=np.int)
    for v1 in g.nodes():
        for v2 in nx.neighbors(g, v1):
            for v3 in nx.neighbors(g, v2):
                if nbnb[int(v1), int(v3)] == 0:
                    v1_nb = set(list(nb1[v1]))
                    v3_nb = set(list(nb1[v3]))
                    intersect = v1_nb.intersection(v3_nb)
                    nbnb[int(v1), int(v3)] = len(intersect)
                for v4 in nx.neighbors(g, v3):
                    if nbnb[int(v1), int(v4)] == 0:
                        v1_nb = set(list(nb2[v1]))
                        v4_nb = set(list(nb2[v4]))
                        intersect = v1_nb.intersection(v4_nb)
                        nbnb[int(v1), int(v4)] = len(intersect)

    return nbnb

def find_common_nb22_v2(g):
    N = g.number_of_nodes()

    nb1 = neighbors1(g)
    nb2 = neighbors2(g)

    nbnb = lil_matrix(np.zeros(shape=(N, N), dtype=np.int), dtype=np.int)
    for v1 in g.nodes():
        for v2 in nx.neighbors(g, v1):
            for v3 in nx.neighbors(g, v2):
                for v4 in nx.neighbors(g, v3):
                    if nbnb[int(v1), int(v4)] == 0:
                        v1_nb = set(list(nb2[v1]))
                        v4_nb = set(list(nb2[v4]))
                        intersect = v1_nb.intersection(v4_nb)
                        nbnb[int(v1), int(v4)] = len(intersect)

    return nbnb

def find_common_nb33(g):
    N = g.number_of_nodes()

    nb1 = neighbors1(g)
    nb2 = neighbors2(g)
    nb3 = neighbors3(g)


    nbnb = lil_matrix(np.zeros(shape=(N, N), dtype=np.int), dtype=np.int)
    for v1 in g.nodes():
        for v2 in nx.neighbors(g, v1):
            for v3 in nx.neighbors(g, v2):
                if nbnb[int(v1), int(v3)] == 0:
                    v1_nb = set(list(nb1[v1]))
                    v3_nb = set(list(nb1[v3]))
                    intersect = v1_nb.intersection(v3_nb)
                    nbnb[int(v1), int(v3)] = len(intersect)
                for v4 in nx.neighbors(g, v3):
                    if nbnb[int(v1), int(v4)] == 0:
                        v1_nb = set(list(nb2[v1]))
                        v4_nb = set(list(nb2[v4]))
                        intersect = v1_nb.intersection(v4_nb)
                        nbnb[int(v1), int(v4)] = len(intersect)
                    for v5 in nx.neighbors(g, v4):
                        if nbnb[int(v1), int(v5)] == 0:
                            v1_nb = set(list(nb3[v1]))
                            v5_nb = set(list(nb3[v5]))
                            intersect = v1_nb.intersection(v5_nb)
                            nbnb[int(v1), int(v5)] = len(intersect)

    return nbnb

def find_common_nb33_v2(g):
    N = g.number_of_nodes()

    nb3 = neighbors3(g)

    nbnb = lil_matrix(np.zeros(shape=(N, N), dtype=np.int), dtype=np.int)
    for v1 in g.nodes():
        for v2 in nx.neighbors(g, v1):
            for v3 in nx.neighbors(g, v2):
                for v4 in nx.neighbors(g, v3):
                    for v5 in nx.neighbors(g, v4):
                        if nbnb[int(v1), int(v5)] == 0:
                            v1_nb = set(list(nb3[v1]))
                            v5_nb = set(list(nb3[v5]))
                            intersect = v1_nb.intersection(v5_nb)
                            nbnb[int(v1), int(v5)] = len(intersect)

    return nbnb




def find_common_nb44(g):
    N = g.number_of_nodes()

    nb1 = neighbors1(g)
    nb2 = neighbors2(g)
    nb3 = neighbors3(g)
    nb4 = neighbors4(g)

    nbnb = lil_matrix(np.zeros(shape=(N, N), dtype=np.int), dtype=np.int)
    for v1 in g.nodes():
        for v2 in nx.neighbors(g, v1):
            for v3 in nx.neighbors(g, v2):
                if nbnb[int(v1), int(v3)] == 0:
                    v1_nb = set(list(nb1[v1]))
                    v3_nb = set(list(nb1[v3]))
                    intersect = v1_nb.intersection(v3_nb)
                    nbnb[int(v1), int(v3)] = len(intersect)
                for v4 in nx.neighbors(g, v3):
                    if nbnb[int(v1), int(v4)] == 0:
                        v1_nb = set(list(nb2[v1]))
                        v4_nb = set(list(nb2[v4]))
                        intersect = v1_nb.intersection(v4_nb)
                        nbnb[int(v1), int(v4)] = len(intersect)
                    for v5 in nx.neighbors(g, v4):
                        if nbnb[int(v1), int(v5)] == 0:
                            v1_nb = set(list(nb3[v1]))
                            v5_nb = set(list(nb3[v5]))
                            intersect = v1_nb.intersection(v5_nb)
                            nbnb[int(v1), int(v5)] = len(intersect)
                        for v6 in nx.neighbors(g, v5):
                            if nbnb[int(v1), int(v6)] == 0:
                                v1_nb = set(list(nb4[v1]))
                                v6_nb = set(list(nb4[v6]))
                                intersect = v1_nb.intersection(v6_nb)
                                nbnb[int(v1), int(v6)] = len(intersect)

    return nbnb

def find_common_nb1(g):
    N = g.number_of_nodes()

    nb_list1 = neighbors1(g)

    nbnb = lil_matrix(np.zeros(shape=(N, N), dtype=np.int), dtype=np.int)
    for v in g.nodes():
        for u in nb_list1[v]:
            v_nb = set(nb_list1[v])
            u_nb = set(nb_list1[u])
            intersect = v_nb.intersection(u_nb)
            nbnb[int(v), int(u)] = len(intersect)

    return nbnb


def find_common_nb2(g):
    N = g.number_of_nodes()

    nb_list2 = neighbors2(g)

    nbnb = lil_matrix(np.zeros(shape=(N, N), dtype=np.int), dtype=np.int)
    for v in g.nodes():
        for u in nb_list2[v]:
            v_nb = set(nb_list2[v])
            u_nb = set(nb_list2[u])
            intersect = v_nb.intersection(u_nb)
            nbnb[int(v), int(u)] = len(intersect)

    return nbnb


def find_common_nb3(g):
    N = g.number_of_nodes()

    nb_list3 = neighbors3(g)

    nbnb = lil_matrix(np.zeros(shape=(N, N), dtype=np.int), dtype=np.int)
    for v in g.nodes():
        for u in nb_list3[v]:
            v_nb = set(nb_list3[v])
            u_nb = set(nb_list3[u])
            intersect = v_nb.intersection(u_nb)
            nbnb[int(v), int(u)] = len(intersect)

    return nbnb

def find_common_nb4(g):
    N = g.number_of_nodes()

    nb_list4 = neighbors4(g)

    nbnb = lil_matrix(np.zeros(shape=(N, N), dtype=np.int), dtype=np.int)
    for v in g.nodes():
        for u in nb_list4[v]:
            v_nb = set(nb_list4[v])
            u_nb = set(nb_list4[u])
            intersect = v_nb.intersection(u_nb)
            nbnb[int(v), int(u)] = len(intersect)

    return nbnb

def find_common_nb5(g):
    N = g.number_of_nodes()

    nb_list5 = neighbors5(g)

    nbnb = lil_matrix(np.zeros(shape=(N, N), dtype=np.int), dtype=np.int)
    for v in g.nodes():
        for u in nb_list5[v]:
            v_nb = set(nb_list5[v])
            u_nb = set(nb_list5[u])
            intersect = v_nb.intersection(u_nb)
            nbnb[int(v), int(u)] = len(intersect)

    return nbnb

def find_common_nb6(g):
    N = g.number_of_nodes()

    nb_list6 = neighbors6(g)

    nbnb = lil_matrix(np.zeros(shape=(N, N), dtype=np.int), dtype=np.int)
    for v in g.nodes():
        for u in nb_list6[v]:
            v_nb = set(nb_list6[v])
            u_nb = set(nb_list6[u])
            intersect = v_nb.intersection(u_nb)
            nbnb[int(v), int(u)] = len(intersect)

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
                score += float(Nvu)*np.log(dot)
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
    common_nb = find_common_nb22(g)

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

        #if iter % 50 == 0:
        #    np.save("../comm/numpy_files/citeseer_poisson_v3_nb3_iter_{}".format(iter), F)
    return F


edges = example1
#g = nx.Graph()
#g.add_edges_from(edges)

g = nx.read_gml("../datasets/karate.gml")


F = run(g, dim=2, num_of_iters=300, eta=0.001)
#np.save("../comm/numpy_files/citeseer_poisson_v3_nb3_son", F)
draw_points(F, "Karate", g)
