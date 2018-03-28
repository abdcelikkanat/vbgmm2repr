import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


ex_graph_0 = [[0,1], [0,2], [1,2], [2,3], [3,4]]
ex_graph_1 = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3], [3,4], [4,5], [4,6], [4,7], [5,6], [5,7], [6,7]]



def show_graph(g):

    plt.figure()
    nx.draw(g, with_labels=True)
    plt.show()

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

def get_nb_list(g):

    nb_list = [[] for _ in range(g.number_of_nodes())]
    for node in g.nodes():
        for nb in nx.neighbors(g, node):
            nb_list[int(node)].append(int(nb))

    return nb_list

def find_eig(g, sample_size=10000):
    """
    A = np.asarray(nx.adjacency_matrix(g).todense(), dtype=np.float)
    P = np.divide(A.T, np.sum(A, 1)).T

    eigval, eigvect = np.linalg.eig(P.T)

    x = eigvect[:, 0].real

    print(x)

    counts = {}
    for i in range(len(x)):
        counts.update({str(i): abs(int(x[i]*sample_size))})

    return counts
    """
    counts = {node: {} for node in g.nodes()}
    for node in g.nodes():
        node_degree = nx.degree(g, node)
        for nb in nx.neighbors(g, node):
            counts[node].update({nb: float((1.0/(float(node_degree)))*float(sample_size))})
            nb_degree = nx.degree(g, nb)
            for nb_nb in nx.neighbors(g, nb):
                if nb_nb not in counts[node] and nb_nb != node:
                    counts[node].update({nb_nb: float((1.0/(float(node_degree)*float(nb_degree)))*float(sample_size))})
                    nb_nb_degree = nx.degree(g, nb_nb)
                    for nb_nb_nb in nx.neighbors(g, nb_nb):
                        if nb_nb_nb not in counts[node] and nb_nb_nb != node:
                            counts[node].update(
                                {nb_nb_nb: float(1.0 / ((float(node_degree) * float(nb_degree) * float(nb_nb_degree))) * float(sample_size))})
                            nb_nb_nb_degree = nx.degree(g, nb_nb_nb)
                            """
                            for nb_nb_nb_nb in nx.neighbors(g, nb_nb_nb):
                                if nb_nb_nb_nb not in counts[node] and nb_nb_nb_nb != node:
                                    counts[node].update(
                                        {nb_nb_nb_nb: float(1.0 / (
                                        (float(node_degree) * float(nb_degree) * float(nb_nb_degree) * float(nb_nb_nb_degree))) * float(
                                            sample_size))})
                            """
                        if nb_nb_nb in counts[node]:
                            counts[node][nb_nb_nb] += float(1.0 / ((float(node_degree) * float(nb_degree) * float(nb_nb_degree))) * float(sample_size))

                if nb_nb in counts[node]:
                    counts[node][nb_nb] += float((1.0 / (float(node_degree) * float(nb_degree))) * float(sample_size))

    return counts

def get_nb_list2(counts):

    nb_list = [[] for _ in range(len(counts))]
    for node in counts:
        for nb in counts[node]:
            nb_list[int(node)].append(int(nb))

    return nb_list



def grad_beta(node, B, T, nb_list, counts):

    grad_sum = 0.0
    for u in nb_list[node]:
        grad_sum += float(counts[str(node)][str(u)]) * (T[u, :] / (1.0 + np.exp(+np.dot(T[u, :], B[node, :]))))

    return grad_sum

def grad_theta(node, B, T, nb_list, counts):

    grad_sum = 0.0
    for v in nb_list[node]:
        grad_sum += float(counts[str(v)][str(node)])*(B[node, :] / (1.0 + np.exp(+np.dot(T[node, :], B[v, :]))))

    return grad_sum

def compute_score(B, T, nb_list, counts):
    N = B.shape[0]

    score = 0.0
    for v in range(N):
        for u in nb_list[v]:
            score += -float(counts[str(v)][str(u)])*np.log(1.0 + np.exp(-np.dot(B[v, :], T[u, :])))

    return score

def run(g, dim, num_of_iters, eta):
    N = nx.number_of_nodes(g)

    # Initialize parameters
    B = np.random.normal(size=(N, dim))
    T = np.random.normal(size=(N, dim))

    T = B

    #B = np.abs(B)
    #T = np.abs(T)



    counts = find_eig(g, sample_size=10000)
    nb_list = get_nb_list2(counts)
    #print(nb_list[1])
    #print(counts[str(1)])
    print("Iteration has just started")
    for iter in range(num_of_iters):
        #if iter % 10 == 0:
        #    draw_points(B, T, "Karate", g, base=True)
        for node in range(N):
            node_grad_B = grad_beta(node, B, T, nb_list, counts)
            for nb in nb_list[node]:
                nb_grad_T = grad_beta(nb, B, T, nb_list, counts)
                T[nb, :] = T[nb, :] + eta*nb_grad_T

            B[node, :] += eta*node_grad_B

        score = compute_score(B, T, nb_list, counts)
        print("Iter: {} Score {}".format(iter, score))

    return B, T


#g = nx.Graph()
#g.add_edges_from(ex_graph_0)
g = nx.read_gml("../datasets/citeseer.gml")


B, T = run(g, dim=128, num_of_iters=1000, eta=0.00001)
np.save("./karate_T", T)
np.save("./karate_B", B)

#draw_points(B, T, name="Karate", g=g)