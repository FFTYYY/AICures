from .gcn import Model as model_gcn
from .gin import Model as model_gin


def get_model_class(C , name):
	return {
		"gcn" : model_gcn , 
		"gin" : model_gin , 
	}[name]