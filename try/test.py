import pysmiles
import dgl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pdb

g = pysmiles.read_smiles("CN1CC[C@@]23C=C[C@@H](C[C@@H]2OC4=C(C=CCC(=C34)C1)OC)O.Br")
#g = pysmiles.read_smiles("O=[N+]([O-])C(Br)(CO)CO")

for i in range(30):
	for j in range(30):
		try:
			e = g[i][j]
		except Exception:
			continue
		print ("%d - %d" % (i,j) , e)

for i in range(min(30 , len(g.nodes))):
	print (g.nodes[i])

pdb.set_trace()
#g = dgl.DGLGraph(g)

def draw(g):
	#g = g.to_networkx().to_undirected()

	def make_color(x):
		if x == 'C':
			return "gray"
		if x == 'N':
			return "blue"
		if x == 'O':
			return "red"
		return "green"

	
	pos = nx.kamada_kawai_layout(g)
	node_type = [g.nodes[i]["element"] for i in range(len(g.nodes))]

	node_labels = nx.get_node_attributes(g, "element")
	nx.draw_networkx_labels(g , pos , labels = node_labels)
	nx.draw(g , pos , node_size = 500 , node_color = [make_color(x) for x in node_type])

	plt.show()

draw(g)
