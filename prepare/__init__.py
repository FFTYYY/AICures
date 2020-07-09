import pdb
import torch as tc
import random
from entry import E
from .dataloader import load_data_tt , load_data_tdt , load_data_files
from .graph_parse import base_parse , augment_dataset
from models import get_model_class
from tqdm import tqdm

def process_graphs(C , dataset):
	lab_num = -1
	
	for i , (g , label , smiles) in tqdm(enumerate(dataset) , ncols = 100 , desc = "Processing..."):

		dgl_g = base_parse(C , g)
		dataset[i] = [dgl_g , label , smiles]

		lab_num = max(lab_num , label) #统计标签数

	return dataset , lab_num + 1

def get_data(C , fold = 0 , pos_lim = -1, neg_lim = -1 , files = False):

	if fold == "test": #读取用于提交的训练-测试集
		trainset , testset = load_data_tt(C , C.data , pos_lim = pos_lim , neg_lim = neg_lim)

		# dev_size = int(len(trainset) * C.dev_prop)
		# devset   = trainset[:dev_size]
		# trainset = trainset[dev_size:]
		devset = []
	elif isinstance(fold , int) : #对 k-fold test，直接读取训练集验证集测试集
		trainset , devset , testset = load_data_tdt(
			C , C.data , 
			path = "train_cv/fold_%d/" % fold , pos_lim = pos_lim , neg_lim = neg_lim
		)
	else: 
		if not files:# fold是某个dataset文件夹的名字
			trainset , devset , testset = load_data_tdt(C , fold , pos_lim = pos_lim , neg_lim = neg_lim)
		else:
			trainset = load_data_files(C , fold , pos_lim = pos_lim , neg_lim = neg_lim)
			devset , testset = [] , []
			
	trainset = augment_dataset(C , trainset)

	trainset , lab_num = process_graphs(C , trainset)
	testset  , _       = process_graphs(C , testset)
	devset   , _ 	   = process_graphs(C , devset)

	return (trainset , devset , testset) , lab_num

def get_model(C , lab_num):
	model = get_model_class(C , C.model)(
		out_d 		  = lab_num , 
		**C.__dict__
	).cuda()

	return model

def get_others(C , model):
	optimer   = tc.optim.Adam(params = model.parameters() , lr = C.lr , weight_decay = C.weight_decay)
	if C.loss_weight:
		loss_func = tc.nn.CrossEntropyLoss(weight = tc.Tensor([0.1 , 1]).cuda())
	else:
		loss_func = tc.nn.CrossEntropyLoss()

	return optimer , loss_func
