import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def get_nb_list(g):
    return [[int(nb) for nb in nx.neighbors(g, str(node))] for node in range(g.number_of_nodes())]

example1 = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [4, 5], [4,6], [4,7], [5,6], [5,7], [6,7], [3,4]]
example1 = [[str(edge[0]), str(edge[1])] for edge in example1]


edges = example1


#g = nx.Graph()
#g.add_edges_from(edges)



"""
plt.figure()
nx.draw(g)
plt.show()
"""


""" ------------------------------
N = g.number_of_nodes()
D = 1
K = 2
eta = 0.01

F = np.random.normal(size=(N, K, D))*1.0
FF = F.copy()
nb_list = get_nb_list(g)

F = np.absolute(F)

epsilon = (2.0*float(g.number_of_edges())) / (float(N) * (float(N)-1.0))
print(epsilon)

logF = 0.0

num_of_iters = 3000
for iter in range(num_of_iters):

    if iter % 100 == 0:
        print("Iter {}, logF {}".format(iter, logF))
    logF = 0.0

    for node in range(N):
        first_term_sum = 0.0
        for nb in nb_list[node]:
            prod = np.sum([np.dot(F[node, k, :], F[nb, k, :]) for k in range(K)])
            first_term_sum += np.log(1.0 - np.exp(-prod) + 1e-6)

        last_term_sum = 0.0
        for lt_nb in range(N):
            if lt_nb != node and lt_nb not in nb_list[node]:
                last_term_sum += np.sum([np.dot(F[node, k, :], F[lt_nb, k, :]) for k in range(K)])

        logF += (first_term_sum - last_term_sum)

    #print(logF)

    for node in range(N):

        first_term = np.zeros(shape=(K, D), dtype=np.float)
        last_term = np.zeros(shape=(K, D), dtype=np.float)
        for nb in nb_list[node]:
            prod = np.sum([np.dot(F[node, k, :], F[nb, k, :]) for k in range(K)])
            #print(prod)
            first_term += F[nb, :, :] * (np.exp(-prod) / (1.0 - np.exp(-prod)+1e-6))

            last_term += F[nb, :, :]

        Fsum = np.sum(F, axis=0)
        last_term = Fsum - last_term - F[node, :, :]

        delta_node = first_term - last_term

        # Update
        F[node, :, :] = F[node, :, :] + eta*delta_node

        FF = F.copy()

        # Make coefficient non-negative
        F[node, :, :] = (F[node, :, :] + np.abs(F[node, :, :])) / 2.0 + 1e-8
"""





""" ------------------------------
print(logF)

edges = [[np.dot(F[node, :, 0], F[nb, :, 0]) for nb in range(N)] for node in range(N) ]
print(np.asarray(edges))


delta = np.sqrt(-np.log(1.0-(1e-8)))
clusters = [[], [], []]
for node in range(N):
    for k in range(K):
        if F[node, k, :] > delta:
            print("Node: {}, Cluster: {} Val: {}".format(node, k, np.sum(F[node, k, :]**1)))
    max_k = np.argmax(F[node, :, 0])
    if F[node, max_k, :] > delta:
            clusters[max_k].append(str(node))

"""

def one_iter(g, F, node):
    # Find F_sum

    F_sum = np.zeros(shape=F.shape[1], dtype=np.float)

    for v in g.nodes():
        if v not in nx.neighbors(g, node):
            F_sum += F[int(v), :]


    # Take derivative wrt F[node, :]
    grad_sum = 0.0
    for v in nx.neighbors(g, node):
        grad_sum += F[int(v), :] * (1.0 / (np.exp(np.dot(F[int(v), :], F[int(node), :])) - 1.0 + 1e-8))
    grad_sum -= F_sum

    return grad_sum

def compute_score(g, F):
    score = 0.0
    for v in g.nodes():
        for u in g.nodes():
            if g.has_edge(v, v):
                score += np.log(1.0 - np.exp(-np.dot(F[int(v), :], F[int(u), :])))
            else:
                score -= np.dot(F[int(v), :], F[int(u), :])
    return score

def find_clusters(g, F, delta, K):
    clusters = [[] for _ in range(K+1)]

    for node in g.nodes():
        k_max = np.argmax(F[int(node), :])
        if F[int(node), k_max] > delta:
            clusters[k_max].append(node)
            #print("Node {}, Cluster {}".format(node, k_max))

    return clusters

def draw_graph(g, clusters):
    K = len(clusters)

    plt.figure()
    pos = nx.spring_layout(g)
    colors = ['r', 'b', 'g', 'b', 'v', 'y', 'p']
    for k in range(K):
        nx.draw_networkx_nodes(g, pos, nodelist=clusters[k],
                               node_color=colors[k],
                               node_size=100,
                               alpha=0.5)
    nx.draw_networkx_labels(g, pos, labels={node: int(node) for node in g.nodes()}, font_size=9)
    nx.draw_networkx_edges(g, pos, width=1.0, alpha=0.5)

    plt.show()

def run():
    dataset = "../datasets/citeseer.gml"
    g = nx.read_gml(dataset)
    N = g.number_of_nodes()
    K = 6
    dim = 1
    eta = 0.0001
    num_of_iters = 500


    # Initialize parameter
    F = np.asarray(np.random.normal(size=(N, K)))
    F = np.absolute(F)

    scores = np.zeros(num_of_iters, dtype=np.float)
    for iter in range(num_of_iters):
        for u in g.nodes():
            grad = one_iter(g, F, u)
            # Update
            F[int(u), :] = F[int(u), :] + eta*grad

            F[int(u), :] = (F[int(u), :] + np.absolute(F[int(u), :])) / 2.0

        scores[iter] = compute_score(g, F)
        print("Iter: {}, Score: {}".format(iter, scores[iter]))
        if iter > 1:
            if abs(scores[iter-1]-scores[iter]) < 0.01:
                break

    #epsilon = (2.0 * g.number_of_edges()) / (N * (N-1))
    epsilon = 1e-8
    delta = np.sqrt(-np.log(1 - epsilon))
    clusters = find_clusters(g, F, delta, K)

    draw_graph(g, clusters)

run()