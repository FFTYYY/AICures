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
				mol = line.strip().split(",")[0]
				label = -1
			elif mode == "train":
				idx , mol , label = line.strip().split(",")

			mol = read_smiles(mol) #将smiles字符串转成networkx graph

			data.append([mol , label])
	return data

def load_data(C):
	path = P.join("data/" , C.data)

	trainset , testset = [load_data_file(P.join(path , p + ".csv") , p) for p in ["train" , "test"]]
	random.shuffle(trainset)

	return trainset , testset