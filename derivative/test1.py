import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def get_nb_list(g):
    return [[int(nb) for nb in nx.neighbors(g, str(node))] for node in range(g.number_of_nodes())]

example1 = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [4, 5], [4,6], [4,7], [5,6], [5,7], [6,7]]


test_g1 = nx.Graph()
test_g1.add_edges_from(example1)

"""
plt.figure()
nx.draw(g)
plt.show()
"""

def sigma(val1, val2):
    return 1.0 / (1.0 + np.exp(-np.dot(val1, val2)))

def update(paths, B, T, eta):
    dim = B.shape[1]

    # Product of sigmas for each length
    sp = [np.prod([sigma(B[path[i], :], T[path[i+1], :]) for i in range(len(path)-1)]) for path in paths]

    total_sp = np.sum(sp)

    t1_sum = np.zeros(shape=dim, dtype=np.float)
    s1_sum = np.zeros(shape=dim, dtype=np.float)
    for i in range(len(paths)):
        path = paths[i]

        t1_sum += sp[i]*(1.0-sigma(B[path[-2], :], T[path[-1], :]))*B[path[-2], :]
        s1_sum += sp[i]*(1.0-sigma(B[path[0], :], T[path[1], :]))*T[path[1], :]

    grad_t = t1_sum / total_sp
    grad_s = s1_sum / total_sp

    # Now update vectors
    s = paths[0][0]
    t = paths[0][-1]
    B[s, :] = grad_s + eta*grad_s
    T[t, :] = grad_t + eta*grad_t


def total_score(paths, B, T):
    n = B.shape[0]

    log_sum = 0.0
    for s in range(n):
        for t in range(n):
            if len(paths[s][t]) > 0:
                path_sums = 0.0
                for p in paths[s][t]:
                    path_sums += np.prod([sigma(B[p[i], :], T[i+1, :]) for i in range(len(p)-1)])

                log_sum += path_sums

    print("Score {}".format(log_sum))

def run():
    g = test_g1
    n = g.number_of_nodes()

    dim = 2
    w = 3
    num_of_paths = 100


    ## Generate paths
    paths = [[[] for _ in range(n)] for _ in range(n)]

    for node in g.nodes():
        for _ in range(num_of_paths):
            p = [int(node)]
            while len(p) < w:
                nb_list = list(nx.neighbors(g, node))

                if len(nb_list) > 0:
                    chosen_node = np.random.choice(a=nb_list, size=1)[0]
                else:
                    chosen_node = node

                p.append(int(chosen_node))

            paths[p[0]][p[-1]].append(p)


    # Initialize vectors
    B = np.random.normal(size=(n, dim))
    T = np.random.normal(size=(n, dim))

    # Update
    num_of_iters = 20
    for _ in range(num_of_iters):
        for s in range(n):
            for t in range(n):
                if len(paths[s][t]) > 0:
                    update(paths=paths[s][t], B=B, T=T, eta=0.001)

        total_score(paths, B, T)
        #print(T)
run()