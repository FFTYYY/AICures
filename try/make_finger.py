import os
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import pdb
from tqdm import tqdm
import pickle

data_path = "../data/pseudomonas/"

fingers = {}
for x in tqdm(os.walk(data_path) , ncols = 120):
	for u in x[2]:
		u = os.path.join(x[0] , u)

		if (not u.endswith(".csv")) or (not "fold" in u):
			continue

		with open(u , "r") as fil:
			fil.readline()
			for line in fil:
				line = line.strip()
				if line == "":
					continue
				smiles , label = line.split(",")
				mol = Chem.MolFromSmiles(smiles)
				finger = list(AllChem.GetMorganFingerprintAsBitVect(mol , 2 , nBits = 1024))

				fingers[smiles] = finger

with open(os.path.join(data_path , "fingers") , "wb") as fil:
	pickle.dump(fingers , fil)

print ("done.")
