import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import dgl
from .base import Base
from dgl import batch , unbatch
from dgl.nn.pytorch.conv import GraphConv
import pdb
from functools import partial

class Model(Base):
	def __init__(self , num_layers , d , out_d , residual , reinit , **kwargs ):

		super().__init__(emb_size = d)

		self.d = d
		self.num_layers = num_layers 
		self.residual = residual

		self.layers = nn.ModuleList([GraphConv(d, d) for _ in range(num_layers)])

		self.ln = nn.Linear(d , out_d)

		if reinit:
			self.reinit_params()

	def reinit_params(self):
		for layer in self.layers:
			nn.init.uniform_(layer.weight , 0 , 1e-5)

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
