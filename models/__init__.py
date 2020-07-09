from .gcn import Model as model_gcn
from .gin import Model as model_gin
from .gat import Model as model_gat
from .egcn import Model as model_egcn
from .rgcn import Model as model_rgcn
from .mix_gcn_lstm import Model as mix_gcn_lstm
import torch as tc


def get_model_class(C , name):
	return {
		"gcn" : model_gcn , 
		"gin" : model_gin , 
		"gat" : model_gat , 
		"egcn" : model_egcn , 
		"rgcn" : model_rgcn , 
		"mix_gcn_lstm" : mix_gcn_lstm , 
	}[name]