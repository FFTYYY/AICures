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
	def __init__(self , num_layers , d , out_d , residual , emb_item_nums , **kwargs ):

		super().__init__(**{x + "_num" : emb_item_nums[x] for x in emb_item_nums} , emb_size = d)

		self.d = d
		self.num_layers = num_layers 
		self.residual = residual

		self.layers = nn.ModuleList([GraphConv(d, d) for _ in range(num_layers)])

		self.ln = nn.Linear(d , out_d)

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
