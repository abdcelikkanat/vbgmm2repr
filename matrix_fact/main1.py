import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def factorize(g, w):
    A = np.asarray(nx.adjacency_matrix(g).todense(), dtype=np.float)
    P = np.divide(A.T, np.sum(A, 1)).T

    R = np.identity(n=P.shape[0], dtype=np.float)
    S = np.zeros(shape=P.shape, dtype=np.float)

    for _ in range(w):
        R = np.dot(P, R)
        S = S + R

    C = 1
    K = np.log(S) + np.log(C)

    eigval, eigvec = np.linalg.eig(K)
    x = eigvec[:, :2].real

    groundTruth = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1]
    plt.figure()
    for i in range(len(groundTruth)):
        if groundTruth[i] == 0:
            plt.plot(x[i, 0], x[i, 1], 'rx')
        if groundTruth[i] == 1:
            plt.plot(x[i, 0], x[i, 1], 'bx')
    plt.show()

g = nx.read_gml("./datasets/karate.gml")
factorize(g, w=5)