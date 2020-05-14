import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import pdb

class Base(nn.Module):
	def __init__(self , element_num , aromatic_num , charge_num , hcount_num , emb_size):
		'''包含基本的 embedding 模块'''
		super().__init__()
		self.emb_elem = nn.Embedding(200 , emb_size)
		self.emb_arom = nn.Embedding(200 , emb_size)
		self.emb_chrg = nn.Embedding(200 , emb_size)
		self.emb_hcnt = nn.Embedding(200 , emb_size)
	
	def get_node_emb(self , g):

		elem_x = self.emb_elem(g.ndata["element" ])
		arom_x = self.emb_arom(g.ndata["aromatic"])
		chrg_x = self.emb_chrg(g.ndata["charge"  ])
		hcnt_x = self.emb_hcnt(g.ndata["hcount"  ])

		x = elem_x + arom_x + chrg_x + hcnt_x

		return x