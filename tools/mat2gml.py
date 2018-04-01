import networkx as nx
import numpy as np
import scipy.io as sio

dataset_name = "blogcatalog"
mat_dict = sio.loadmat("../mat_files/"+dataset_name+".mat")
print(mat_dict['network'].shape)

n = mat_dict['network'].shape[1]
c = mat_dict['group'].shape[1]


g = nx.Graph()
networkx = mat_dict['network']
cx = networkx.tocoo()

for i in range(len(cx.row)):
    if cx.data[i]:
        g.add_edge(str(cx.row[i]), str(cx.col[i]))

print(g.number_of_nodes())
print(g.number_of_edges())


g = nx.Graph()
network = mat_dict['network']
cx = network.tocoo()
for i, j, val in zip(cx.row, cx.col, cx.data):
    if val:
        g.add_edge(str(i), str(j))



group = mat_dict['group']

cluster = {}
for i in range(n):
    cluster.update({str(i): np.argmax(group[i])})

nx.set_node_attributes(g, values=cluster, name='clusters')

nx.write_gml(g, "../datasets/"+dataset_name+".gml")





