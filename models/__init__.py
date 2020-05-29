from .gcn import Model as model_gcn
from .gin import Model as model_gin
from .gat import Model as model_gat
from .egcn import Model as model_egcn
from .rgcn import Model as model_rgcn

def get_model_class(C , name):
	return {
		"gcn" : model_gcn , 
		"gin" : model_gin , 
		"gat" : model_gat , 
		"egcn" : model_egcn , 
		"rgcn" : model_rgcn , 
	}[name]