import torch as th
from torch import nn
from torch.nn import init
import torch.nn.functional as F

import torch as tc
import dgl
import pdb
from dgl import batch , unbatch


from dgl import function as fn
from dgl.base import DGLError

class RConv(nn.Module):
	def __init__(self , in_feats , out_feats , norm="both"):
		super(RConv, self).__init__()

		self._in_feats = in_feats
		self._out_feats = out_feats
		self._norm = norm

		self.o_emb = nn.Embedding(10 , in_feats * out_feats)
		self.bias = nn.Parameter(tc.rand(out_feats))
		
		self.reset_parameters()

	def reset_parameters(self):
		init.xavier_uniform_(self.o_emb.weight)
		init.zeros_(self.bias)

	def norm_opt_before(self , graph , feat):
		if self._norm == "both":
			degs = graph.out_degrees().to(feat.device).float().clamp(min=1)
			norm = th.pow(degs, -0.5)
			shp = norm.shape + (1,) * (feat.dim() - 1)
			norm = th.reshape(norm, shp)
			feat = feat * norm
		return feat

	def norm_opt_after(self , graph , feat , rst):
		if self._norm != "none":
			degs = graph.in_degrees().to(feat.device).float().clamp(min=1)
			norm = th.pow(degs, -0.5) # if norm == "both"
			shp = norm.shape + (1,) * (feat.dim() - 1)
			norm = th.reshape(norm, shp)
			rst = rst * norm
		return rst

	def my_message_func(self , edges):

		o_w = self.o_emb(edges.data["o"]).view(-1 , self._out_feats , self._in_feats)
		h = edges.src["h"].view(-1 , self._in_feats , 1)

		h = tc.matmul(o_w , h).view(-1 , self._out_feats)

		return {
			"h_m" : h,
		}

	def my_reduce_func(self , nodes):

		h_m = nodes.mailbox["h_m"]
		h_m = h_m.sum(dim = 1)

		return {"h" : h_m}


	def forward(self , graph , feat):

		graph = graph.local_var()

		feat = self.norm_opt_before(graph , feat)
		
		graph.srcdata["h"] = feat
		graph.edata["o"] = graph.edata["order"]

		graph.register_message_func(self.my_message_func)
		graph.register_reduce_func(self.my_reduce_func)	

		graph.send(graph.edges())
		graph.recv(graph.nodes())
		
		x = graph.dstdata["h"]
		x = self.norm_opt_after(graph , feat , x)
		x = x + self.bias

		return x


from .base import Base as Base

class Model(Base):
	def __init__(self , num_layers , d , out_d , residual , reinit , **kwargs ):

		super().__init__(emb_size = d)

		self.d = d
		self.num_layers = num_layers 
		self.residual = residual

		self.layers = nn.ModuleList([RConv(d, d) for _ in range(num_layers)])

		self.ln = nn.Linear(d , out_d)

		if reinit:
			self.reinit_params()

	def reinit_params(self):
		for layer in self.layers:
			nn.init.uniform_(layer.o_emb.weight , 0 , 1e-5)

	def forward(self , gs):

		g = batch(gs)
		x = self.get_node_emb(g)

		for i , layer in enumerate(self.layers):
			old_x = x
			x = F.relu(layer(g , x))
			if self.residual:
				x = x + old_x

		x = self.ln(x)

		g.ndata["x"] = x
		gs = unbatch(g)

		xs = [g.ndata["x"] for g in gs]
		xs = [x.mean(dim = 0 , keepdim = True) for x in xs]
		xs = tc.cat(xs , dim = 0)

		return xs
