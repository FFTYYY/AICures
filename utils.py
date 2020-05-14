import torch as tc

def copy_param(tar , src):
	'''把src的参数copy给tar'''

	tar_param = list(tar.parameters())
	src_param = list(src.parameters())
	
	assert len(tar_param) == len(src_param)
	
	for i in range(len(src_param)):
		with tc.no_grad():
			tar_param[i].data = src_param[i].data.clone().detach()
	return tar