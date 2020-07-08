import torch as tc
import os
import pickle

def copy_param(tar , src):
	'''把src的参数copy给tar'''

	tar_param = list(tar.parameters())
	src_param = list(src.parameters())
	
	assert len(tar_param) == len(src_param)
	
	for i in range(len(src_param)):
		with tc.no_grad():
			tar_param[i].data = src_param[i].data.clone().detach()
	return tar


def save_model(to_save , save_path , exp_id , run_name):
	p = os.path
	save_p = p.join(save_path , str(exp_id) , str(run_name) + ".pkl")
	os.makedirs(p.dirname(save_p) , exist_ok =  True)
	with open(save_p , "wb") as fil:
		pickle.dump(to_save , fil)