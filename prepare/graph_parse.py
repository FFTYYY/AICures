from dgl import DGLGraph
import torch as tc

eles = {}

def get_element_num():
	return len(eles)

def ele2num(e):
	if eles.get(e) is None:
		eles[e] = len(eles)
	return eles[e]

def base_parse(g):
	'''将networkx graph转成DGLGraph'''

	g = DGLGraph()

	feat_elem = [ ele2num(g.nodes[i]["element"]) for i in range(len(g.nodes))]
	feat_elem = tc.LongTensor(feat_elem)

	g.ndata["element"] = feat_elem

	return g