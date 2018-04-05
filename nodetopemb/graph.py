import random
import node2vec
import networkx as nx
import numpy as np
import scipy.sparse as scio

class Graph:
    """

    """
    adj_list = []
    num_of_nodes = 0
    num_of_edges = 0

    def __init__(self):
        pass

    def number_of_nodes(self):

        return self.num_of_nodes

    def number_of_edges(self):

        return self.num_of_edges

    def add_edges_from(self, edge_list):
        # It is assumed that node labels starts from 0 and up to n

        self.num_of_nodes = max([max(edge) for edge in edge_list]) + 1

        self.adj_list = [[] for _ in range(self.num_of_nodes)]
        for edge in edge_list:
            self.adj_list[edge[0]].append(edge[1])
            self.adj_list[edge[1]].append(edge[0])

            # Increase number of edges by 1 since it is a simple graph
            self.num_of_edges += 1


    def nb_list(self, node):
        return self.adj_list[node]


    def deepwalk_step(self, path_length, alpha=0.0, rand=random.Random(), starting_node=None):

        if starting_node is None:
            starting_node = rand.choice(range(self.number_of_nodes()))

        path = [starting_node]
        current_path_length = 1

        while current_path_length < path_length:
            # Get the latest appended node
            latest_node = path[-1]
            # If the current node has any neighbour
            if len(self.adj_list[latest_node]) > 0:
                # Return to the starting node with probability alpha
                if rand.random() >= alpha:
                    path.append(rand.choice(self.adj_list[latest_node]))
                else:
                    path.append(path.append(self.adj_list[0]))

                current_path_length += 1
            else:
                break

        return path


    def count_triangles_on_edges(self):
        edgeId = 0
        triCountOnEdges = scio.lil_matrix((self.num_of_nodes, self.num_of_nodes), dtype=np.int)

        for u in range(self.num_of_nodes):
                for v in self.adj_list[u]:
                    if u < v:
                        for w in self.adj_list[v]:
                            if v < w:
                                for x in self.adj_list[w]:
                                    if x == u:
                                        triCountOnEdges[u, v] += 1
                                        triCountOnEdges[v, w] += 1
                                        triCountOnEdges[w, u] += 1

                                        triCountOnEdges[v, u] += 1
                                        triCountOnEdges[w, v] += 1
                                        triCountOnEdges[u, w] += 1
        return triCountOnEdges


    def triangle_walk_step(self, path_length, alpha=0.0, rand=random.Random(), starting_node=None, triangle_count=None):

        if starting_node is None:
            starting_node = rand.choice(range(self.number_of_nodes()))

        path = [starting_node]
        current_path_length = 1

        while current_path_length < path_length:
            # Get the latest appended node
            latest_node = path[-1]
            # If the current node has any neighbour
            if len(self.adj_list[latest_node]) > 0:
                # Return to the starting node with probability alpha
                if rand.random() >= alpha:
                    prob = [float(triangle_count[latest_node, node]) for node in self.adj_list[latest_node]]
                    norm_constant = np.sum(prob)
                    if norm_constant == 0.0:
                        prob = [1.0/float(len(self.adj_list[latest_node])) for _ in range(len(self.adj_list[latest_node]))]
                    else:
                        prob = prob / norm_constant
                    path.append(np.random.choice(self.adj_list[latest_node], p=prob))
                else:
                    path.append(path.append(self.adj_list[0]))

                current_path_length += 1
            else:
                break

        return path


    def degreeBasedWalk_step(self, path_length, alpha=0.0, rand=random.Random(), starting_node=None):

        if starting_node is None:
            starting_node = rand.choice(range(self.number_of_nodes()))

        path = [starting_node]
        current_path_length = 1

        while current_path_length < path_length:
            # Get the latest appended node
            latest_node = path[-1]

            # If the current node has any neighbour
            if len(self.adj_list[latest_node]) > 0:
                # Return to the starting node with probability alpha
                if rand.random() >= alpha:
                    path.append(rand.choice(self.adj_list[latest_node]))

                    weights = np.asarray([float(len(self.adj_list[nb]))**2 for nb in self.adj_list[latest_node]])
                    weights = weights / np.sum(weights)
                    chosen_nb = np.random.choice(a=self.adj_list[latest_node], size=1, p=weights)[0]
                    path.append(chosen_nb)

                else:
                    path.append(path[0])

                current_path_length += 1
            else:
                break

        return path


    def contUpdatedWalk_step(self, path_length, alpha=0.0, rand=random.Random(), starting_node=None, weights=None):

        if weights is None:
            weights = [[1 for _ in self.adj_list[i]] for i in range(self.number_of_nodes())]

        if starting_node is None:
            starting_node = rand.choice(range(self.number_of_nodes()))

        path = [starting_node]
        current_path_length = 1

        while current_path_length < path_length:
            # Get the latest appended node
            latest_node = path[-1]
            # If the current node has any neighbour
            if len(self.adj_list[latest_node]) > 0:
                # Return to the starting node with probability alpha
                if rand.random() >= alpha:
                    prob = [float(weights[latest_node][i]) for i in range(len(self.adj_list[latest_node]))]
                    norm_const = np.sum(prob)
                    if norm_const == 0.0:
                        prob = [1.0/len(prob) for _ in range(len(prob))]
                    else:
                        prob = prob / norm_const
                    chosen_node_inx = np.random.choice(range(len(self.adj_list[latest_node])), p=prob)
                    chosen_node = self.adj_list[latest_node][chosen_node_inx]
                    path.append(chosen_node)
                    weights[latest_node][chosen_node_inx] += 1
                    # weights[chosen_node][latest_node] += 1
                else:
                    path.append(path.append(self.adj_list[0]))

                current_path_length += 1
            else:
                break

        return path


    def graph2doc(self, number_of_paths, path_length, params=dict(), rand=random.Random(), method="Deepwalk"):

        corpus = []

        node_list = range(self.num_of_nodes)

        if method == "Deepwalk":
            alpha = params['alpha']

            for _ in range(number_of_paths):
                # Shuffle the nodes
                rand.shuffle(node_list)
                # For each node, initialize a random walk
                for node in node_list:
                    walk = self.deepwalk_step(path_length=path_length, rand=rand, alpha=alpha,
                                              starting_node=node)
                    corpus.append(walk)

        if method == "Node2Vec":
            # Generate the desired networkx graph
            p = params['p']
            q = params['q']

            nxg = nx.Graph()
            for i in range(self.number_of_nodes()):
                for j in self.adj_list[i]:
                    nxg.add_edge(str(i), str(j))
                    nxg[str(i)][str(j)]['weight'] = 1

            G = node2vec.Graph(nx_G=nxg, p=p, q=q, is_directed=False)
            G.preprocess_transition_probs()
            walks = G.simulate_walks(num_walks=number_of_paths, walk_length=path_length)

            corpus = walks

        if method == "TriWalk":
            alpha = params['alpha']

            tri_count = 0.03 * scio.csr_matrix(self.count_triangles_on_edges())

            for _ in range(number_of_paths):
                # Shuffle the nodes
                rand.shuffle(node_list)
                # For each node, initialize a random walk
                for node in node_list:
                    walk = self.triangle_walk_step(path_length=path_length, rand=rand, alpha=alpha,
                                                   starting_node=node, triangle_count=tri_count)
                    corpus.append(walk)

        if method == "contUpdatedWalk":
            alpha = params['alpha']
            weights = params['weights']
            if weights is None:
                weights = [[len(self.adj_list[j]) for j in self.adj_list[i]] for i in range(self.number_of_nodes())]

            for _ in range(number_of_paths):
                # Shuffle the nodes
                rand.shuffle(node_list)
                # For each node, initialize a random walk
                for node in node_list:
                    walk = self.contUpdatedWalk_step(path_length=path_length, rand=rand, alpha=alpha,
                                                     starting_node=node, weights=weights)
                    corpus.append(walk)

            params['weights'] = weights

        if method == "degreeBasedWalk":
            alpha = params['alpha']

            for _ in range(number_of_paths):
                # Shuffle the nodes
                rand.shuffle(node_list)
                # For each node, initialize a random walk
                for node in node_list:
                    walk = self.degreeBasedWalk_step(path_length=path_length, alpha=alpha,
                                                     rand=random.Random(), starting_node=node)

                    corpus.append(walk)

        return corpus
