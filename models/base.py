import torch as tc
import torch.nn as nn
import torch.nn.functional as F

class Base(nn.Module):
	def __init__(self , element_num , emb_size):
		'''包含基本的 embedding 模块'''
		super().__init__()
		self.emb_elem = nn.Embedding(element_num , emb_size)
	
	def get_node_emb(self , g):
		elem_x = self.emb_elem(g.ndata["element"])
		return elem_x