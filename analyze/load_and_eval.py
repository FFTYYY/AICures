import os
import pdb
import argparse
from fitterlog.interface import new_or_load_experiment
import pickle
from YTools.experiment_helper import set_random_seed
from prepare import get_data , get_others
from train_procedure.evaluate import evaluate
from train_procedure.train import train
import torch as tc
import re

par = argparse.ArgumentParser()
par.add_argument("--data" 		, type = str , default = "pseudomonas")
par.add_argument("--seed" 		, type = int , default = 2333)
par.add_argument("--save_path" 	, type = str , default = "save_model/")
par.add_argument("--save_name" 	, type = str , default = "750/7")
C = par.parse_args()
E = new_or_load_experiment(project_name = "PRML" , group_name = "analyze")
if C.seed > 0:
	set_random_seed(C.seed)

E.new_variable("Dev ROC-AUC")
E.new_variable("Dev PRC-AUC")
E.new_variable("Test ROC-AUC")
E.new_variable("Test PRC-AUC")
E.new_variable("Train Loss")
E.new_variable("Dev Loss")
E.new_variable("Test Loss")

def main():

	with open(os.path.join(C.save_path , C.save_name + ".pkl") , "rb") as fil:
		model = pickle.load(fil)

	run_id 		= int(re.search("/(\\d+)$" , C.save_name).group(1))
	loss_func 	= tc.nn.CrossEntropyLoss()
	optimer   	= tc.optim.Adam(params = model.parameters() , lr = 1e-3 , weight_decay = 1e-8)
	C.uniform_sample = True
	C.grad_clip = -1 
	C.bs 		= 10
	(trainset , devset , testset) , lab_num = get_data  (C , fold = run_id)

	model , train_loss  = train(C, model, trainset, loss_func, optimer, 0, run_id, 0)
	print(train_loss)

	troc_auc , tprc_auc = evaluate(C, model, testset , loss_func, 0, run_id, 0, "Test")
	print(troc_auc , tprc_auc)
	pdb.set_trace()
	troc_auc , tprc_auc = evaluate(C, model, testset , loss_func, 0, run_id, 0, "Test")
	troc_auc , tprc_auc = evaluate(C, model, testset , loss_func, 0, run_id, 0, "Test")
	troc_auc , tprc_auc = evaluate(C, model, testset , loss_func, 0, run_id, 0, "Test")

	#print(droc_auc , dprc_auc)
	#print(rroc_auc , rprc_auc)


if __name__ == "__main__":
	main()

	E.finish()