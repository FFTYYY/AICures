from .gcn import Model as model_gcn


def get_model_class(C , name):
	return {
		"gcn" : model_gcn , 
	}[name]