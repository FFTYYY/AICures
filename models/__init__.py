from .gcn import Model as model_gcn
from .gin import Model as model_gin
from .gat import Model as model_gat

def get_model_class(C , name):
	return {
		"gcn" : model_gcn , 
		"gin" : model_gin , 
		"gat" : model_gat , 
	}[name]