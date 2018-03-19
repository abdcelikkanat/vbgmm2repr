import numpy as np
import networkx as nx


g = nx.read_gml("../datasets/karate.gml")


output_edge_file = "../temp/" + "output.txt"

with open(output_edge_file, 'w') as f:
    f.write("#line1\n#line2\n#line3\n#line4")
    for node in range(g.number_of_nodes()):
        for nb in range(g.number_of_nodes()):
            if g.has_edge(str(node), str(nb)) and nb >= node:
                f.write("\n{}\t{}".format(node, nb))

