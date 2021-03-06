import numpy as np
import networkx as nx


def get_number_of_clusters(g):

    cluster = nx.get_node_attributes(g, "clusters")

    max_label = 0
    for node in g.nodes():
        if int(cluster[node]) > max_label:
            max_label = int(cluster[node])

    return max_label + 1


def degree_sequence(g, sorted=False):
    deg_seq = []
    for node in g.nodes():
        deg_seq.append(nx.degree(g, node))

    if sorted is True:
        deg_seq.sort()
        deg_seq = deg_seq[::-1]

    return deg_seq


def cluster(g, deep=1):

    N = g.number_of_nodes()
    num_of_clusters = get_number_of_clusters(g)
    cluster = nx.get_node_attributes(g, "clusters")

    node_cluster_counts = [[0 for _ in range(num_of_clusters)] for _ in range(N)]

    nb_list = {node: [] for node in g.nodes()}

    for node in g.nodes():
        for nb in nx.neighbors(g, node):
            if nb not in nb_list[node]:
                nb_list[node].append(nb)
                node_cluster_counts[int(node)][int(cluster[nb])] += 1
            if deep > 1:
                for nb_nb in nx.neighbors(g, nb):
                    if nb_nb not in nb_list[node]:
                        nb_list[node].append(nb_nb)
                        node_cluster_counts[int(node)][int(cluster[nb_nb])] += 1
                    if deep > 2:
                        for nb_nb_nb in nx.neighbors(g, nb_nb):
                            if nb_nb_nb not in nb_list[node]:
                                nb_list[node].append(nb_nb_nb)
                                node_cluster_counts[int(node)][int(cluster[nb_nb_nb])] += 1
                                if deep > 3:
                                    for nb_nb_nb_nb in nx.neighbors(g, nb_nb_nb):
                                        if nb_nb_nb_nb not in nb_list[node]:
                                            nb_list[node].append(nb_nb_nb_nb)
                                            node_cluster_counts[int(node)][int(cluster[nb_nb_nb_nb])] += 1

    classification = 0
    for node in g.nodes():
        if cluster[node] == np.argmax(node_cluster_counts[int(node)]):
            classification += 1

    return float(classification)*100.0 / float(N)




g = nx.read_gml("../datasets/blogcatalog.gml")

#d = degree_sequence(g, True)
#print(d)
nb_clusters = cluster(g, deep=3)
print(nb_clusters)
print(get_number_of_clusters(g))