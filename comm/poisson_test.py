import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random


def get_nb_list(g):
    return [[int(nb) for nb in nx.neighbors(g, str(node))] for node in range(g.number_of_nodes())]

example1 = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [4, 5], [4,6], [4,7], [5,6], [5,7], [6,7], [3,4]]
example1 = [[str(edge[0]), str(edge[1])] for edge in example1]

example2 = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [4, 5], [4,6], [4,7], [5,6], [5,7], [6,7]]
example2 = [[str(edge[0]), str(edge[1])] for edge in example2]

example3 = [[0, 1], [0, 2], [0, 3], [0, 4], [5, 6], [5, 7], [5, 8], [5, 9], [0, 5]]
example3 = [[str(edge[0]), str(edge[1])] for edge in example3]

edges = example3


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

def one_iter(g, B, T, eta):

    C = np.zeros(shape=B.shape, dtype=np.float)
    D = np.zeros(shape=T.shape, dtype=np.float)

    for node in range(N):
        delta_T = 0.0
        for s in range(N):
            check = 0
            if g.has_edge(str(node), str(s)):
                check = 1
            delta_T += check * ( B[s, :] / (np.exp(np.dot(B[s, :], T[node, :])) - 1.0)) + (1-check) * (-B[s, :])

        C[node, :] = T[node, :] + eta*delta_T



    for node in range(N):
        delta_B = 0.0
        for t in range(N):
            check = 0
            if g.has_edge(str(node), str(t)):
                check = 1
            delta_B += check * (T[t, :] / ( np.exp(np.dot(B[node, :], T[t, :])) - 1.0)) + (1-check) * (-T[t, :])

        D[node, :] = B[node, :] + eta*delta_B

    return D, C

def one_iter2(g, B, T, eta):
    C = np.zeros(shape=B.shape, dtype=np.float)
    for node in range(N):
        delta_B = 0.0
        for s in range(N):
            check = 0
            if g.has_edge(str(node), str(s)):
                check = 1
            if node != s:
                delta_B += check * ( B[s, :] / (np.exp(np.dot(B[s, :], B[node, :])) - 1.0)) + (1-check) * (-B[s, :])

        B[node, :] = B[node, :] + eta*delta_B
        B = np.abs(B)

    return B, T

def log_score(g, B, T):
    n = g.number_of_nodes()
    score = 0.0
    for i in range(n):
        for j in range(i+1, n):
            check = 0
            if g.has_edge(str(i), str(j)):
                check = 1
            score += check*np.log(1.0 - np.exp(-np.dot(B[i, :],T[j, :])) + 1e-08) + (1-check)*(-np.dot(B[i, :],T[j, :]))

    return score

def log_score2(g, B, T):
    n = g.number_of_nodes()
    score = 0.0
    for i in range(n):
        for j in range(i+1, n):
            check = 0
            if g.has_edge(str(i), str(j)):
                check = 1
            score += check*np.log(1.0 - np.exp(-np.dot(B[i, :], B[j, :])) + 1e-08) + (1-check)*(-np.dot(B[i, :],B[j, :]))

    return score

def run():
    dim = 2
    num_of_iters = 2000
    n = g.number_of_nodes()
    eta = 0.001

    # Initialize
    B = np.random.normal(size=(n, dim))
    T = np.random.normal(size=(n, dim))

    B = np.abs(B)
    T = np.abs(T)

    scores = np.zeros(shape=num_of_iters, dtype=np.float)
    for iter in range(num_of_iters):
        B, T = one_iter2(g, B, T, eta)
        scores[iter] = log_score2(g, B, T)
        print("{} Score: {}".format(iter, scores[iter]))

    for i in range(n):
        for j in range(i+1,n):
            s = np.dot(B[i, :], B[j,:])
            print("Edge {}-{} : {}".format(i, j, s))

def derive_target(g, S, T, sinx, tinx, node):

    grad = 0.0
    for s in sinx:
        normalization = np.sum([np.exp(np.dot(S[s, :], T[k, :])) for k in tinx]) + 1e-08
        for t in tinx:
            if g.has_edge(str(s), str(t)):
                check = 1 if node == t else 0
                grad += S[s, :]*( check - (np.exp(np.dot(S[s, :], T[node, :])) / normalization) )

    return grad

def derive_base(g, S, T, sinx, tinx, node):

    grad = 0.0
    for s in sinx:
        normalization = np.sum([np.exp(np.dot(S[s, :], T[k, :])) for k in tinx]) + 1e-08
        for t in tinx:
            if g.has_edge(str(s), str(t)):
                check = 1 if node == s else 0
                grad += T[t, :]*( check - (np.exp(np.dot(S[node, :], T[t, :])) / normalization) )

    return grad


def softmax_iter(g, S, T, sinx, tinx, pairs, eta):
    n = g.number_of_nodes()

    # For each 'node', take derivative

    for pair in pairs:

        s = pair[0]
        t = pair[1]
        if t in [1,2,3,4] and s == 0:
            t_grad = derive_target(g, S, T, sinx=[0], tinx=[1,2,3,4], node=t)
            s_grad = derive_base(g, S, T, sinx=[0], tinx=[1,2,3,4], node=s)

            T[t, :] = T[t, :] + eta * t_grad
            S[s, :] = S[s, :] + eta * s_grad

        if t in [6, 7, 8, 9]  and s == 5:
            t_grad = derive_target(g, S, T, sinx=[5], tinx=[6, 7, 8, 9], node=t)
            s_grad = derive_base(g, S, T, sinx=[5], tinx=[6, 7, 8, 9], node=s)

            T[t, :] = T[t, :] + eta * t_grad
            S[s, :] = S[s, :] + eta * s_grad
    """
    nodes = [i for i in range(g.number_of_nodes())]
    random.shuffle(nodes)
    for node in nodes:
        S[t, :] = S[t, :] + eta * s_grad
    """

    return S, T

def score(S, T, sinx, tinx):
    score = 0.0
    for s in sinx:
        normalizer = np.sum([np.exp(np.dot(S[s, :], T[k, :])) for k in tinx])
        for t in tinx:
            if g.has_edge(str(s), str(t)):
                score += np.dot(S[s, :], T[t, :]) - np.log(normalizer)

    return score

def run2():
    n = g.number_of_nodes()
    dim = 2

    num_of_iters = 100
    eta = 0.01
    sinx = [0, 5]
    #tinx = [i for i in range(n)]
    tinx = [1,2,3,4,6,7,8,9]

    # Initialize parameters
    S = np.random.normal(size=(n, dim))
    T = np.random.normal(size=(n, dim))

    pairs = [[i, int(j)] for i in [0, 5] for j in nx.neighbors(g, str(i))]
    pairs.remove([0, 5])
    pairs.remove([5, 0])
    pairs.extend(pairs)
    pairs.extend(pairs)
    pairs.extend(pairs)

    print(pairs)

    random.shuffle(pairs)
    for iter in range(num_of_iters):
        S, T = softmax_iter(g, S, T, sinx, tinx, pairs, eta)
        #S, T = softmax_iter(g, S, T, sinx, tinx, pairs2, eta)
        sc = score(S, T, sinx, tinx)
        print("Score: {}".format(sc))


    for i in range(n):
        for j in range(n):
            s = np.exp(np.dot(S[i, :], T[j, :]))
            print("{}-{} : {}".format(i, j, s))
    """
    plt.figure()
    plt.plot(T[:4, 0], T[:4, 1], 'r.')
    plt.plot(T[4:, 0], T[4:, 1], 'b.')
    plt.plot(S[:4, 0], S[:4, 1], 'rx')
    plt.plot(S[4:, 0], S[4:, 1], 'bx')
    plt.show()
    """
    plt.figure()
    plt.plot(T[1:5, 0], T[1:5, 1], 'r.')
    plt.plot(T[6:, 0], T[6:, 1], 'b.')
    plt.plot(S[0, 0], S[0, 1], 'rx')
    plt.plot(S[5, 0], S[5, 1], 'bx')
    plt.plot(T[0, 0], T[0, 1], 'ro')
    plt.plot(T[5, 0], T[5, 1], 'bo')
    plt.show()

# run()
run2()