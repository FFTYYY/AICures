import pdb
import torch as tc
import random
from config import E
from .dataloader import load_data
from .graph_parse import base_parse , get_element_num
from models import get_model_class

def process_graphs(dataset):
	lab_num = -1	
	for i , (g , label) in enumerate(dataset):

		dgl_g = base_parse(g)
		dataset[i] = [dgl_g , label]

		label = int(label)

		lab_num = max(lab_num , label) #统计标签数

	return dataset , lab_num + 1

def get_data(C):

	trainset , testset = load_data(C)

	trainset , lab_num = process_graphs(trainset)
	testset  , _       = process_graphs(testset)

	dev_size = int(len(trainset) * C.dev_prop)
	devset   = trainset[:dev_size]
	trainset = trainset[dev_size:]

	return (trainset , devset , testset) , lab_num

def get_model(C , lab_num):
	model = get_model_class(C , C.model)(
		num_layers 	= C.num_layers , 
		d 			= C.d , 
		out_d 		= lab_num , 
		element_num = get_element_num() , 
	).cuda()

	return model

def get_others(C , model):
	optimer   = tc.optim.Adam(params = model.parameters() , lr = C.lr)
	loss_func = tc.nn.CrossEntropyLoss()

	return optimer , loss_func
