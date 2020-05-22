from dgl import DGLGraph
import torch as tc

eles = {}

def ele2num(e):
	if eles.get(e) is None:
		eles[e] = len(eles)
	return eles[e]

def base_parse(g):
	'''将networkx graph转成DGLGraph'''

	feat_elem = [ ele2num(g.nodes[i]["element"]) for i in range(len(g.nodes))]
	feat_elem = tc.LongTensor(feat_elem)
	feat_arom = tc.LongTensor([int(g.nodes[i]["aromatic"]) for i in range(len(g.nodes))])
	feat_chrg = tc.LongTensor([int(g.nodes[i]["charge"  ]) for i in range(len(g.nodes))]) + 20 #防止负数
	feat_hcnt = tc.LongTensor([int(g.nodes[i]["hcount"  ]) for i in range(len(g.nodes))])


	g = DGLGraph(g)
	g.ndata["element" ] = feat_elem.cuda()
	g.ndata["aromatic"] = feat_arom.cuda()
	g.ndata["charge"  ] = feat_chrg.cuda()
	g.ndata["hcount"  ] = feat_hcnt.cuda()


	g.add_edges(g.nodes() , g.nodes()) #添加自环

	return g