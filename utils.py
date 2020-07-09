import torch as tc
import os
import pickle
import pdb

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

def load_model(save_path , exp_id , run_name):
	p = os.path
	save_p = p.join(save_path , str(exp_id) , str(run_name) + ".pkl")
	os.makedirs(p.dirname(save_p) , exist_ok =  True)
	with open(save_p , "rb") as fil:
		model = pickle.load(fil)
	return model

def save_pred(preds , data , save_name):
	p = os.path
	with open(p.join("data" , data , "test.csv") , "r") as fil_in , \
		open(p.join("data" , data , save_name) , "w") as fil_out:

		fil_out.write(fil_in.readline())
		i = 0
		for line in fil_in:
			line = line.strip()
			if line == "":
				fil_out.write("\n")
			else:
				fil_out.write(line + str(preds[i]) + "\n")
				i = i + 1

class EnsembleModel:
	'''不可用于训练'''
	def __init__(self , models):
		self.models = models
		self.softmaxed = True #不用再softmax

	def __call__(self , *pargs , **kwargs):

		results = []
		for model in self.models:
			model = model.eval()

			with tc.no_grad():
				results.append( tc.softmax(model(*pargs , **kwargs) , dim = -1) )

		result = 0
		for res in results:
			result = result + res
		result /= len(results)

		return result

	def eval(self):
		return self
