import os , sys
import os.path as P
from pysmiles import read_smiles
import random
import pdb
import time

def load_data_file(path , mode):
	
	stdout_backup = sys.stdout
	fake_stdout = open("tmp_%d.txt" % (int(time.time())) , "w")
	sys.stdout = fake_stdout
	print ("stdout not shut down correctly")

	data = []

	with open(path , "r") as fil:
		fil.readline()
		for i , line in enumerate(fil):
			if mode == "test":
				mol , label 	= line.strip().split(",")[0] , -1
			elif mode == "train":
				_ , mol , label = line.strip().split(",")
			elif mode == "tdt":
				mol , label 	= line.strip().split(",")

			mol = read_smiles(mol) #将smiles字符串转成networkx graph

			data.append([mol , int(label)])

			if i % 1000 == 0:
				sys.stderr.write("%d\n" % i)

	fake_stdout.close()
	sys.stdout = stdout_backup
	print ("stdout recoverd.")

	return data

def load_data_tt(C , data): # train and test
	path = P.join("data/" , data)

	trainset , testset = [load_data_file(P.join(path , p + ".csv") , p) for p in ["train" , "test"]]
	random.shuffle(trainset)

	return trainset , testset

def load_data_tdt(C , data , path = ""): # train , dev & test
	path = P.join("data/" , data , path)

	trainset , devset , testset = [load_data_file(P.join(path , p + ".csv") , "tdt") for p in 
		["train" , "dev" , "test"]
	]
	random.shuffle(trainset)

	return trainset , devset , testset