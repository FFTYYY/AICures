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
from .egcn import RConv
from .base import Base as Base

class LSTMModel(nn.Module):
	def __init__(self , input_size , hidden_size , num_layers , bidrect , dropout):
		super().__init__()

		if num_layers <= 1:
			dropout = 0.0
		
		self.rnn = nn.LSTM(input_size = input_size , hidden_size = hidden_size , 
			num_layers = num_layers , batch_first = True , dropout = dropout , 
			bidirectional = bidrect)


		self.number = (2 if bidrect else 1) * num_layers

	def forward(self , x , lens):
		'''
			mask : (bs) 
			x : (bs , sl , is)
		'''
		lens , idx_sort = tc.sort(lens , descending = True)
		_ , idx_unsort = tc.sort(idx_sort)

		x = x[idx_sort]
		
		x = nn.utils.rnn.pack_padded_sequence(x , lens , batch_first = True)
		self.rnn.flatten_parameters()
		y , (h , c) = self.rnn(x)
		y , lens = nn.utils.rnn.pad_packed_sequence(y , batch_first = True)

		h = h.transpose(0,1).contiguous() #make batch size first

		y = y[idx_unsort]		#(bs , seq_len , bid * hid_size)
		h = h[idx_unsort]		#(bs , number , hid_size)

		return y , h


class Process_Smiles(nn.Module):
	def __init__(self , d):
		super().__init__()

		self.emb = nn.Embedding(1024 , d , padding_idx = 0)
		self.lstm = LSTMModel(d , d , 2 , True , 0.0)

		self.ln = nn.Linear(self.lstm.number * d , d)

	def forward(self , smiles):
		lens = tc.LongTensor([len(x) for x in smiles]).cuda()

		max_len = int(max(lens))
		smiles = [[ord(c) for c in x] + [0 for _ in range(max_len - len(x))] for x in smiles]
		x = tc.LongTensor(smiles).cuda()
		x = self.emb(x)

		y , h = self.lstm(x , lens)
		h = h.view(h.size(0) , -1)
		h = self.ln(h) #(bs , d)
		return h

class Model(Base):
	def __init__(self , num_layers , d , out_d , residual , reinit , layer_norm , dropout , activate , **kwargs ):

		super().__init__(emb_size = d)

		self.d 			= d
		self.num_layers = num_layers 
		self.residual 	= residual
		self.layer_norm = layer_norm
		self.activate 	= activate

		self.lstm = Process_Smiles(d)
		self.str_ln = nn.ModuleList([nn.Linear(2 * d , d) for _ in range(num_layers)])

		self.layers = nn.ModuleList([RConv(d, d) for _ in range(num_layers)])
		if layer_norm:
			self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(num_layers)])
		self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])

		self.ln = nn.Linear(d , out_d)

		if reinit:
			self.reinit_params()

	def reinit_params(self):
		for layer in self.layers:
			nn.init.uniform_(layer.o_emb.weight , 0 , 1e-5)

	def forward(self , gs , smiles , **kwargs):

		d = self.d

		str_h = self.lstm(smiles) #(bs , d)，字符串特征
		for i in range(len(str_h)):
			gs[i].ndata["str_h"] = str_h[i].view(1 , d).expand(gs[i].number_of_nodes() , d)

		g = batch(gs)
		x = self.get_node_emb(g)
		str_h = g.ndata["str_h"]

		for i , layer in enumerate(self.layers):
			old_x = x

			x = layer(g , x)

			if self.activate == "relu":
				x = F.relu(x)
			elif self.activate == "leaky":
				x = F.leaky_relu(x , 0.1)

			if self.layer_norm:
				x = self.norms[i](x)

			x = F.relu(self.str_ln[i](tc.cat([x , str_h] , dim = -1)))

			if self.residual:
				x = x + old_x

			x = self.dropout[i](x)

		x = self.ln(x)

		g.ndata["x"] = x
		gs = unbatch(g)

		xs = [g.ndata["x"] for g in gs]
		xs = [x.mean(dim = 0 , keepdim = True) for x in xs]
		xs = tc.cat(xs , dim = 0)

		return xs
