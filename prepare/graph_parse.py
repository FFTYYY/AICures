from dgl import DGLGraph
import torch as tc

eles = {}

def ele2num(e):
	if eles.get(e) is None:
		eles[e] = len(eles)
	return eles[e]

def base_parse(g):
	'''将networkx graph转成DGLGraph'''

	g = g.to_directed()
	feat_elem = [ ele2num(g.nodes[i]["element"]) for i in range(len(g.nodes))]
	feat_elem = tc.LongTensor(feat_elem)
	feat_arom = tc.LongTensor([int(g.nodes[i]["aromatic"]) for i in range(len(g.nodes))])
	feat_chrg = tc.LongTensor([int(g.nodes[i]["charge"  ]) for i in range(len(g.nodes))]) + 20 #防止负数
	feat_hcnt = tc.LongTensor([int(g.nodes[i]["hcount"  ]) for i in range(len(g.nodes))])
	feat_ordr = tc.LongTensor([int(g[u][v]["order"] * 2) for u,v in g.edges])

	dg = DGLGraph(g)
	dg.ndata["element" ] = feat_elem.cuda()
	dg.ndata["aromatic"] = feat_arom.cuda()
	dg.ndata["charge"  ] = feat_chrg.cuda()
	dg.ndata["hcount"  ] = feat_hcnt.cuda()
	dg.edata["order"   ] = feat_ordr.cuda()

	dg.add_edges(dg.nodes() , dg.nodes()) #添加自环

	return dg