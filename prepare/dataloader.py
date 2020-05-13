import os , sys
import os.path as P
from pysmiles import read_smiles
import random
import pdb

def load_data_file(path , mode):
	
	data = []

	with open(path , "r") as fil:
		fil.readline()
		for line in fil:
			if mode == "test":
				mol , label 	= line.strip().split(",")[0] , -1
			elif mode == "train":
				_ , mol , label = line.strip().split(",")
			elif mode == "kfold":
				mol , label 	= line.strip().split(",")

			mol = read_smiles(mol) #将smiles字符串转成networkx graph

			data.append([mol , int(label)])
	return data

def load_data_tt(C): # train and test
	path = P.join("data/" , C.data)

	trainset , testset = [load_data_file(P.join(path , p + ".csv") , p) for p in ["train" , "test"]]
	random.shuffle(trainset)

	return trainset , testset

def load_data_tdt(C , path): # train , dev & test
	path = P.join("data/" , C.data , path)

	trainset , devset , testset = [load_data_file(P.join(path , p + ".csv") , "kfold") for p in 
		["train" , "dev" , "test"]
	]
	random.shuffle(trainset)

	return trainset , devset , testset