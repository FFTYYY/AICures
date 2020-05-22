import os , sys
import os.path as P
from pysmiles import read_smiles
import random
import pdb
import time

def myp(x = ""):
	sys.stderr.write(str(x) + "\n")
	sys.stderr.flush()

def load_data_file(path , mode , pos_lim = -1 , neg_lim = -1):
	
	stdout_backup = sys.stdout
	sys.stdout = open(os.devnull , "w")
	print ("stdout not shut down correctly")

	num_pos = 0
	num_neg = 0

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

			label = int(label)

			if label >= 0:
				if label == 0:
					num_neg += 1
					if neg_lim > 0 and num_neg > neg_lim:
						continue
				else:
					num_pos += 1
					if pos_lim > 0 and num_pos > pos_lim:
						continue
			if (pos_lim > 0 and num_pos > pos_lim) and (neg_lim > 0 and num_neg > neg_lim):
				break

			mol = read_smiles(mol) #将smiles字符串转成networkx graph

			data.append([mol , int(label)])

			if i % 1000 == 0:
				sys.stderr.write("%d\n" % i)
	sys.stdout = stdout_backup
	print ("stdout recoverd.")

	return data

def load_data_tt(C , data , pos_lim = -1 , neg_lim = -1): # train and test
	path = P.join("data/" , data)

	trainset , testset = [
		load_data_file(P.join(path , p + ".csv") , p , pos_lim , neg_lim) 
		for p in ["train" , "test"]
	]
	random.shuffle(trainset)

	return trainset , testset

def load_data_tdt(C , data , path = "" , pos_lim = -1 , neg_lim = -1): # train , dev & test
	path = P.join("data/" , data , path)

	trainset , devset , testset = [
		load_data_file(P.join(path , p + ".csv") , "tdt" , pos_lim , neg_lim ) 
		for p in ["train" , "dev" , "test"]
	]
	random.shuffle(trainset)

	return trainset , devset , testset

def load_data_files(C , files = "" , pos_lim = -1, neg_lim = -1 , mode = "tdt" ):

	trainset = []
	for file in files:
		path = P.join("data/" , file)

		trainset += load_data_file(path , mode , pos_lim , neg_lim ) 
	random.shuffle(trainset)

	return trainset